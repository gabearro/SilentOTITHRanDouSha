use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProtocolError {
    #[error("invalid parameters: {0}")]
    InvalidParams(String),

    #[error("field arithmetic error: {0}")]
    FieldError(String),

    #[error("shamir sharing error: {0}")]
    ShamirError(String),

    #[error("silent OT error: {0}")]
    SilentOtError(String),

    #[error("randousha error: {0}")]
    RanDouShaError(String),

    #[error("multiplication error: {0}")]
    MultiplyError(String),

    #[error("network error: {0}")]
    NetworkError(String),

    #[error("serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    #[error("verification failed: {0}")]
    VerificationFailed(String),

    #[error("malicious party detected: {0}")]
    MaliciousParty(String),
}

pub type Result<T> = std::result::Result<T, ProtocolError>;
