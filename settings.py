from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My FastAPI Application"
    environment: str = "development"
    database_url: str

    class Config:
        env_file = ".env"
