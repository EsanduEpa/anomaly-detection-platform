from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # These must match your .env file exactly
    DATABASE_URL: str
    REDIS_URL: str
    APP_NAME: str = "Anomaly Detection Platform"
    DEBUG: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"  # Ignore any .env variables not defined above
    )

settings = Settings()