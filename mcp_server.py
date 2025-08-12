# mcp_server.py
import os
import uuid
import logging
import asyncio
import chromadb
import uvicorn
import git
import docker
from io import StringIO
from pylint.reporters.text import TextReporter
from pylint.pyreverse.main import Run as PyLintRun
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# --- Konfigurace ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("agent_server.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- Inicializace ---
app = FastAPI(title="AI Super Tým pro Copilota")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)
try:
    client = chromadb.HttpClient(host='localhost', port=8000)
    memory_collection = client.get_or_create_collection(name="copilot_super_team_memory")
    logger.info("Úspěšně připojeno k ChromaDB.")
except Exception as e:
    logger.error(f"Nepodařilo se připojit k ChromaDB: {e}.")
    client = None


# --- JÁDRA SPECIALIZOVANÝCH AGENTŮ ---
async def run_specialist_agent(agent_name: str, system_prompt: str, task_description: str) -> str:
    """Obecná funkce pro spuštění specializovaného agenta."""
    logger.info(f"{agent_name.upper()}: Přijal úkol: '{task_description}'")
    messages = [("system", system_prompt), ("human", task_description)]
    response = await llm.ainvoke(messages)
    content = response.content.strip()

    # Očištění kódu od značek pro kódovací agenty
    if agent_name in ["coder_backend", "coder_frontend", "db_specialist"]:
        code_types = ["python", "html", "css", "javascript", "sql"]
        for code_type in code_types:
            content = content.replace(f"```{code_type}", "").replace("```", "")
        content = content.strip()

    return content


# --- NÁSTROJE PRO MANAŽERA ---
async def write_file(path: str, content: str) -> str:
    logger.info(f"MANAŽER (Nástroj): Zapisuji do souboru: '{path}'")
    try:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory): os.makedirs(directory)
        with open(path, 'w', encoding='utf-8') as f:
            await asyncio.to_thread(f.write, content)
        return f"Soubor '{path}' byl úspěšně uložen."
    except Exception as e:
        return f"Chyba při zápisu do souboru: {e}"


async def read_file(path: str) -> str:
    logger.info(f"MANAŽER (Nástroj): Čtu soubor: '{path}'")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return await asyncio.to_thread(f.read)
    except Exception as e:
        return f"Chyba při čtení souboru: {e}"


async def run_in_sandbox_terminal(command: str) -> str:
    logger.info(f"MANAŽER (Terminál): Spouštím příkaz: '{command}'")
    try:
        docker_client = docker.from_env()
        container = docker_client.containers.get('my-sandbox')
        exit_code, output = await asyncio.to_thread(container.exec_run, command)
        decoded_output = output.decode('utf-8').strip()
        return f"Příkaz dokončen (kód: {exit_code}).\nVýstup:\n---\n{decoded_output}"
    except docker.errors.NotFound:
        return "Chyba: Sandbox kontejner 'my-sandbox' neběží."
    except Exception as e:
        return f"Chyba při komunikaci s Dockerem: {e}"


# --- API ENDPOINTS ---
class QueryInput(BaseModel): query: str


class WriteFileInput(BaseModel): path: str; content: str


class DelegateInput(BaseModel):
    agent_name: str = Field(...,
                            description="Jméno specialisty: 'innovator', 'architect', 'coder_backend', 'coder_frontend', 'db_specialist', 'qa_tester'.")
    task_description: str


@app.get("/health")
async def health_check():
    """Jednoduchý endpoint pro Docker health check."""
    return {"status": "ok"}


@app.post("/delegate_task")
async def delegate_task_endpoint(data: DelegateInput):
    prompts = {
        "innovator": "Jsi 'Principal Engineer' a expert na kreativní řešení. Navrhni 2-3 inovativní přístupy k danému problému. U každého stručně popiš jeho princip, výhody a nevýhody.",
        "architect": "Jsi softwarový architekt. Tvým úkolem je vzít zadání a rozbít ho na detailní, technický plán kroků. Odpověz POUZE jako číslovaný seznam.",
        "coder_backend": "Jsi expert na Python. Tvým úkolem je napsat čistý, funkční kód podle zadání. Vrať POUZE kód.",
        "coder_frontend": "Jsi expert na frontend. Napiš čistý HTML, CSS a JavaScript kód podle zadání. Vrať POUZE kód.",
        "db_specialist": "Jsi databázový specialista. Napiš efektivní SQL dotaz nebo schéma podle zadání. Vrať POUZE SQL kód.",
        "qa_tester": "Jsi QA Tester. Tvým úkolem je zkontrolovat kvalitu Python kódu v daném souboru pomocí Pylint."
    }
    agent_name = data.agent_name.lower()
    if agent_name not in prompts:
        raise HTTPException(status_code=400, detail="Neznámé jméno agenta.")

    # QA tester je speciální, protože pracuje se souborem, ne s popisem
    if agent_name == "qa_tester":
        if not os.path.exists(data.task_description):
            return f"Chyba: Soubor '{data.task_description}' neexistuje."
        string_io = StringIO()
        reporter = TextReporter(string_io)
        await asyncio.to_thread(PyLintRun, [data.task_description], reporter=reporter, exit=False)
        output = string_io.getvalue()
        result = "Kontrola kvality proběhla úspěšně. Kód je čistý." if len(
            output.splitlines()) <= 2 else f"Nalezeny problémy:\n{output}"
    else:
        result = await run_specialist_agent(agent_name, prompts[agent_name], data.task_description)

    return {"result": result}


@app.post("/write_file")
async def write_file_endpoint(data: WriteFileInput): return {"result": await write_file(data.path, data.content)}


@app.post("/read_file")
async def read_file_endpoint(data: QueryInput): return {"result": await read_file(data.query)}


@app.post("/run_in_terminal")
async def terminal_endpoint(data: QueryInput): return {"result": await run_in_sandbox_terminal(data.query)}


if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8001)
