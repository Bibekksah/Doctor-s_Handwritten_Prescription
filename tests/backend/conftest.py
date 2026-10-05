import pytest
import httpx

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.database import Base, get_db
from backend.database.models import Medicine
from backend.main import app


@pytest.fixture
def test_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=engine)

    try:
        yield engine
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def test_session(test_engine):
    TestingSessionLocal = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )

    db = TestingSessionLocal()

    try:
        db.add(
            Medicine(
                name="Aceta",
                generic_name="Paracetamol",
            )
        )
        db.commit()

        yield db
    finally:
        db.close()


@pytest.fixture
async def client(test_session):
    def override_get_db():
        yield test_session

    app.dependency_overrides[get_db] = override_get_db

    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            yield client
    finally:
        app.dependency_overrides.clear()