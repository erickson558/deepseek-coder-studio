from fastapi.testclient import TestClient

from app.core.version import VERSION
from app.main import app


def test_health_endpoint() -> None:
    """El endpoint /health devuelve status ok y la versión actual del proyecto.

    Se usa la constante VERSION importada para que el test no quede desincronizado
    al hacer bumps de versión futuros.
    """
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    # Versión dinámica: no hardcodear para que el test siga pasando tras cada bump.
    assert payload["version"] == VERSION
