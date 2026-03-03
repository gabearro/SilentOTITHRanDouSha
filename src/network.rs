use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

pub struct PartyNetwork {
    pub party_id: usize,
    pub n: usize,
    senders: HashMap<usize, mpsc::Sender<(usize, Vec<u8>)>>,
    receiver: mpsc::Receiver<(usize, Vec<u8>)>,
}

impl PartyNetwork {
    pub async fn send_to<T: Serialize>(&self, to: usize, msg: &T) -> Result<(), String> {
        let data = bincode::serialize(msg).map_err(|e| e.to_string())?;
        self.senders
            .get(&to)
            .ok_or_else(|| format!("no channel to party {}", to))?
            .send((self.party_id, data))
            .await
            .map_err(|e| e.to_string())
    }

    pub async fn broadcast<T: Serialize>(&self, msg: &T) -> Result<(), String> {
        let data = bincode::serialize(msg).map_err(|e| e.to_string())?;
        for sender in self.senders.values() {
            sender
                .send((self.party_id, data.clone()))
                .await
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub async fn recv<T: for<'de> Deserialize<'de>>(&mut self) -> Result<(usize, T), String> {
        let (from, data) = self.receiver.recv().await.ok_or("channel closed")?;
        let msg: T = bincode::deserialize(&data).map_err(|e| e.to_string())?;
        Ok((from, msg))
    }

    pub async fn recv_from_all<T: for<'de> Deserialize<'de>>(
        &mut self,
    ) -> Result<HashMap<usize, T>, String> {
        let mut results = HashMap::new();
        for _ in 0..self.n - 1 {
            let (from, msg) = self.recv::<T>().await?;
            results.insert(from, msg);
        }
        Ok(results)
    }
}

pub fn setup_channel_network(n: usize) -> Vec<PartyNetwork> {
    let mut incoming_txs: Vec<mpsc::Sender<(usize, Vec<u8>)>> = Vec::with_capacity(n);
    let mut incoming_rxs: Vec<Option<mpsc::Receiver<(usize, Vec<u8>)>>> = Vec::with_capacity(n);

    for _ in 0..n {
        let (tx, rx) = mpsc::channel(10_000);
        incoming_txs.push(tx);
        incoming_rxs.push(Some(rx));
    }

    let mut networks = Vec::with_capacity(n);
    for i in 0..n {
        let mut senders = HashMap::new();
        for j in 0..n {
            if i != j {
                senders.insert(j, incoming_txs[j].clone());
            }
        }
        networks.push(PartyNetwork {
            party_id: i,
            n,
            senders,
            receiver: incoming_rxs[i].take().unwrap(),
        });
    }

    networks
}

pub async fn setup_tcp_network(n: usize, base_port: u16) -> Vec<PartyNetwork> {
    let mut listeners = Vec::with_capacity(n);
    for i in 0..n {
        let port = base_port + i as u16;
        let listener = TcpListener::bind(format!("127.0.0.1:{}", port))
            .await
            .unwrap_or_else(|e| panic!("failed to bind port {}: {}", port, e));
        listeners.push(listener);
    }

    let mut incoming_txs: Vec<mpsc::Sender<(usize, Vec<u8>)>> = Vec::with_capacity(n);
    let mut incoming_rxs: Vec<Option<mpsc::Receiver<(usize, Vec<u8>)>>> = Vec::with_capacity(n);
    for _ in 0..n {
        let (tx, rx) = mpsc::channel(10_000);
        incoming_txs.push(tx);
        incoming_rxs.push(Some(rx));
    }

    let mut outgoing_txs: Vec<HashMap<usize, mpsc::Sender<Vec<u8>>>> =
        (0..n).map(|_| HashMap::new()).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            let port_j = base_port + j as u16;
            let client_stream = TcpStream::connect(format!("127.0.0.1:{}", port_j))
                .await
                .unwrap();
            let (server_stream, _) = listeners[j].accept().await.unwrap();

            let (client_read, client_write) = client_stream.into_split();
            let (server_read, server_write) = server_stream.into_split();

            let (tx_ij, mut rx_ij) = mpsc::channel::<Vec<u8>>(10_000);
            outgoing_txs[i].insert(j, tx_ij);
            tokio::spawn(async move {
                let mut writer = client_write;
                while let Some(data) = rx_ij.recv().await {
                    let len = (data.len() as u32).to_le_bytes();
                    if writer.write_all(&len).await.is_err() {
                        break;
                    }
                    if writer.write_all(&data).await.is_err() {
                        break;
                    }
                }
            });

            let incoming_j = incoming_txs[j].clone();
            tokio::spawn(async move {
                let mut reader = server_read;
                loop {
                    let mut len_buf = [0u8; 4];
                    if reader.read_exact(&mut len_buf).await.is_err() {
                        break;
                    }
                    let len = u32::from_le_bytes(len_buf) as usize;
                    if len > MAX_MESSAGE_SIZE {
                        break;
                    }
                    let mut data = vec![0u8; len];
                    if reader.read_exact(&mut data).await.is_err() {
                        break;
                    }
                    if incoming_j.send((i, data)).await.is_err() {
                        break;
                    }
                }
            });

            let (tx_ji, mut rx_ji) = mpsc::channel::<Vec<u8>>(10_000);
            outgoing_txs[j].insert(i, tx_ji);
            tokio::spawn(async move {
                let mut writer = server_write;
                while let Some(data) = rx_ji.recv().await {
                    let len = (data.len() as u32).to_le_bytes();
                    if writer.write_all(&len).await.is_err() {
                        break;
                    }
                    if writer.write_all(&data).await.is_err() {
                        break;
                    }
                }
            });

            let incoming_i = incoming_txs[i].clone();
            tokio::spawn(async move {
                let mut reader = client_read;
                loop {
                    let mut len_buf = [0u8; 4];
                    if reader.read_exact(&mut len_buf).await.is_err() {
                        break;
                    }
                    let len = u32::from_le_bytes(len_buf) as usize;
                    if len > MAX_MESSAGE_SIZE {
                        break;
                    }
                    let mut data = vec![0u8; len];
                    if reader.read_exact(&mut data).await.is_err() {
                        break;
                    }
                    if incoming_i.send((j, data)).await.is_err() {
                        break;
                    }
                }
            });
        }
    }

    let mut networks = Vec::with_capacity(n);
    for i in 0..n {
        let mut senders = HashMap::new();
        for (&j, tcp_tx) in &outgoing_txs[i] {
            let (adapter_tx, mut adapter_rx) = mpsc::channel::<(usize, Vec<u8>)>(10_000);
            let tcp_tx_clone = tcp_tx.clone();
            tokio::spawn(async move {
                while let Some((_from, data)) = adapter_rx.recv().await {
                    if tcp_tx_clone.send(data).await.is_err() {
                        break;
                    }
                }
            });
            senders.insert(j, adapter_tx);
        }
        networks.push(PartyNetwork {
            party_id: i,
            n,
            senders,
            receiver: incoming_rxs[i].take().unwrap(),
        });
    }

    networks
}
