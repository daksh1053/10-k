import os
import logging
from copy import deepcopy
from typing import Dict, List, Optional

from flask import (
    Flask,
    flash,
    redirect,
    render_template_string,
    request,
    url_for,
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from settings import DEFAULT_CONFIG
from store.vector_store import VectorStore
from store.document_processing import TextPreProcessor, TablePreProcessor
from store.persistent import PersistentStore
from chain.rag_chain import RAGChain
from agents.react_agent import ReactAgent
from agents.react_with_reflection import ReactWithReflection


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
app.config["UPLOAD_FOLDER"] = os.path.join("app_data", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# ----------------------------------------------------------------------
# Global application state
# ----------------------------------------------------------------------
persistent_store = PersistentStore()
app_config: Dict = deepcopy(DEFAULT_CONFIG)
app_config["huggingface_token"] = os.environ.get("HF_API_TOKEN", "")

if not app_config["huggingface_token"]:
    raise RuntimeError("HF_API_TOKEN environment variable is required for embeddings.")

vector_store = VectorStore(app_config)

# ----------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------
BASE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>10-K Knowledge Workbench</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f6f7; margin: 0; padding: 0; color: #222; }
        header { background: #1f3c88; color: #fff; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; }
        header h1 { margin: 0; font-size: 20px; }
        header nav a { color: #fff; margin-left: 16px; text-decoration: none; font-weight: 500; }
        .container { max-width: 1200px; margin: 24px auto; padding: 0 16px; }
        .workspace { display: flex; gap: 24px; align-items: flex-start; }
        .sidebar { width: 220px; display: flex; flex-direction: column; gap: 12px; }
        .nav-button { display: block; padding: 12px 16px; border-radius: 10px; background: #e7ecf8; color: #1f3c88; text-decoration: none; font-weight: 600; transition: background 0.2s ease, color 0.2s ease; }
        .nav-button:hover { background: #d4dff5; }
        .nav-button.active { background: #1f3c88; color: #fff; }
        .content { flex: 1; }
        .module { background: #fff; border-radius: 12px; box-shadow: 0 8px 24px rgba(31, 60, 136, 0.08); padding: 24px; }
        h2 { margin-top: 0; color: #1f3c88; }
        h3 { margin-top: 24px; color: #0d1b2a; }
        form { display: grid; gap: 12px; margin-top: 12px; }
        label { font-weight: 600; }
        input[type="text"], textarea, select { padding: 10px; border-radius: 8px; border: 1px solid #cfd8e3; font-size: 14px; width: 100%; box-sizing: border-box; }
        textarea { min-height: 120px; resize: vertical; }
        input[type="file"] { font-size: 14px; }
        button { background: #1f3c88; color: #fff; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 600; width: fit-content; }
        button:hover { background: #162c62; }
        ul { padding-left: 18px; margin: 12px 0; }
        .flash { background: #ffefc1; color: #8a6200; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; }
        .result { margin-top: 24px; background: #0d2538; color: #f1f5f9; border-radius: 12px; padding: 18px; }
        .result pre { white-space: pre-wrap; font-family: "Courier New", monospace; line-height: 1.4; }
        .history-preview { background: #f1f5ff; border-radius: 8px; padding: 12px 16px; margin-top: 18px; }
        .history-preview h4 { margin: 0 0 8px 0; color: #1f3c88; }
        .history-item { margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #d6e0f5; }
        .history-item:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
        .badge { display: inline-block; background: #1f3c88; color: #fff; border-radius: 999px; padding: 2px 10px; font-size: 12px; margin-right: 8px; text-transform: uppercase; letter-spacing: 0.3px; }
        .filing-list { background: #f8fafc; border-radius: 8px; padding: 12px 16px; margin-top: 18px; }
        .filing-entry { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #e2e8f0; }
        .filing-entry:last-child { border-bottom: none; }
        .empty { color: #5f6c7b; font-style: italic; }
        .history-full { display: grid; gap: 16px; margin-top: 16px; }
        .history-full-entry { background: #0d2538; color: #f1f5f9; border-radius: 12px; padding: 16px; }
        .history-full-entry pre { white-space: pre-wrap; font-family: "Courier New", monospace; line-height: 1.4; margin-top: 12px; }
        .meta { color: #5f6c7b; font-size: 13px; margin-top: 6px; }
        @media (max-width: 960px) {
            .workspace { flex-direction: column; }
            .sidebar { width: 100%; flex-direction: row; flex-wrap: wrap; }
            .nav-button { flex: 1 1 30%; text-align: center; }
        }
    </style>
</head>
<body>
    <header>
        <h1>10-K Knowledge Workbench</h1>
        <nav>
            <a href="{{ url_for('index', view='ingestion') }}">Ingestion</a>
            <a href="{{ url_for('index', view='query') }}">Query</a>
            <a href="{{ url_for('index', view='history') }}">History</a>
        </nav>
    </header>
    <div class="container">
        {% with messages = get_flashed_messages() %}
        {% if messages %}
            <div class="flash">
                {% for message in messages %}
                    <div>{{ message }}</div>
                {% endfor %}
            </div>
        {% endif %}
        {% endwith %}
        <div class="workspace">
            <aside class="sidebar">
                <a class="nav-button {% if active_view == 'ingestion' %}active{% endif %}" href="{{ url_for('index', view='ingestion') }}">Ingestion</a>
                <a class="nav-button {% if active_view == 'query' %}active{% endif %}" href="{{ url_for('index', view='query') }}">Query</a>
                <a class="nav-button {% if active_view == 'history' %}active{% endif %}" href="{{ url_for('index', view='history') }}">History</a>
            </aside>
            <main class="content">
                {% if active_view == 'ingestion' %}
                    <section class="module">
                        <h2>Filing &amp; Document Ingestion</h2>
                        <form action="{{ url_for('add_filing') }}" method="post">
                            <h3>Create Filing from URLs</h3>
                            <div>
                                <label for="filing_name">Filing Name</label>
                                <input id="filing_name" name="filing_name" type="text" placeholder="e.g. Tesla 2024 10-K" required>
                            </div>
                            <div>
                                <label for="filing_urls">Source URLs (one per line)</label>
                                <textarea id="filing_urls" name="filing_urls" placeholder="https://www.example.com/filing"></textarea>
                            </div>
                            <button type="submit">Ingest Filing</button>
                        </form>
                        <form action="{{ url_for('upload_pdf') }}" method="post" enctype="multipart/form-data">
                            <h3>Upload PDF Tables</h3>
                            <div>
                                <label for="pdf_filing_id">Assign to Filing</label>
                                <select id="pdf_filing_id" name="filing_id" required>
                                    <option value="" disabled selected>Select a filing</option>
                                    {% for filing in filings %}
                                        <option value="{{ filing.id }}">{{ filing.name }} (ID {{ filing.id }})</option>
                                    {% endfor %}
                                </select>
                            </div>
                            <div>
                                <label for="pdf_file">PDF File</label>
                                <input id="pdf_file" type="file" name="pdf_file" accept="application/pdf" required>
                            </div>
                            <button type="submit">Extract Tables</button>
                        </form>
                        <h3>Stored Filings</h3>
                        <div class="filing-list">
                            {% if filings %}
                                {% for filing in filings %}
                                    <div class="filing-entry">
                                        <span><strong>{{ filing.name }}</strong></span>
                                        <span class="badge">ID {{ filing.id }}</span>
                                    </div>
                                {% endfor %}
                            {% else %}
                                <div class="empty">No filings ingested yet.</div>
                            {% endif %}
                        </div>
                    </section>
                {% elif active_view == 'query' %}
                    <section class="module">
                        <h2>Query Module</h2>
                        <form action="{{ url_for('run_query') }}" method="post">
                            <div>
                                <label for="query_text">Question</label>
                                <textarea id="query_text" name="query_text" required>{{ query_text or "" }}</textarea>
                            </div>
                            <div>
                                <label for="query_mode">Mode</label>
                                <select id="query_mode" name="query_mode">
                                    <option value="decomposer" {% if query_mode == 'decomposer' %}selected{% endif %}>Query Decomposer (RAG)</option>
                                    <option value="react" {% if query_mode == 'react' %}selected{% endif %}>ReAct Agent</option>
                                    <option value="reflexion" {% if query_mode == 'reflexion' %}selected{% endif %}>Reflexion Agent</option>
                                </select>
                            </div>
                            <div>
                                <label for="filing_ids">Target Filings (hold Ctrl or Cmd to select multiple)</label>
                                <select id="filing_ids" name="filing_ids" multiple size="6">
                                    {% for filing in filings %}
                                        <option value="{{ filing.id }}" {% if filing.id in selected_filings %}selected{% endif %}>{{ filing.name }} (ID {{ filing.id }})</option>
                                    {% endfor %}
                                </select>
                            </div>
                            <button type="submit">Run Query</button>
                        </form>
                        {% if query_result %}
                            <div class="result">
                                <h3>Answer · {{ query_mode_label }}</h3>
                                <pre>{{ query_result }}</pre>
                            </div>
                        {% endif %}
                        <div class="history-preview">
                            <h4>Recent Queries</h4>
                            {% if history_preview %}
                                {% for item in history_preview %}
                                    <div class="history-item">
                                        <div><span class="badge">{{ item.mode }}</span>{{ item.query }}</div>
                                        <small>Filings: {{ ", ".join(item.filing_labels) if item.filing_labels else "All" }}</small>
                                    </div>
                                {% endfor %}
                                <a href="{{ url_for('index', view='history') }}">View full history</a>
                            {% else %}
                                <div class="empty">No queries yet.</div>
                            {% endif %}
                        </div>
                    </section>
                {% elif active_view == 'history' %}
                    <section class="module">
                        <h2>Query History</h2>
                        {% if history_full %}
                            <div class="history-full">
                                {% for item in history_full %}
                                    <div class="history-full-entry">
                                        <div><span class="badge">{{ item.mode }}</span>{{ item.query }}</div>
                                        <div class="meta">Filings: {{ ", ".join(item.filing_labels) if item.filing_labels else "All" }} · {{ item.timestamp }}</div>
                                        <pre>{{ item.answer }}</pre>
                                    </div>
                                {% endfor %}
                            </div>
                        {% else %}
                            <div class="empty">No queries recorded yet.</div>
                        {% endif %}
                    </section>
                {% endif %}
            </main>
        </div>
    </div>
</body>
</html>
"""


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _get_filing_labels(ids: Optional[List[int]]) -> List[str]:
    if not ids:
        return []
    filings = {filing["id"]: filing["name"] for filing in persistent_store.list_filings()}
    return [f"{filings.get(fid, 'Unknown')} (ID {fid})" for fid in ids]


def _render_dashboard(
    query_result: Optional[str] = None,
    query_mode: str = "decomposer",
    query_text: Optional[str] = None,
    selected_filings: Optional[List[int]] = None,
    active_view: str = "ingestion",
) -> str:
    filings_data = persistent_store.list_filings()
    filings = [type("Filing", (), filing) for filing in filings_data]

    history_raw = persistent_store.get_query_history()
    history_preview = [
        {
            "mode": item["mode"],
            "query": item["query"],
            "filing_labels": _get_filing_labels(item.get("filing_ids")),
        }
        for item in history_raw[:5]
    ]

    history_full = []
    if active_view == "history":
        for item in history_raw:
            history_full.append(
                {
                    "mode": item["mode"],
                    "query": item["query"],
                    "answer": item["answer"],
                    "timestamp": item["timestamp"],
                    "filing_labels": _get_filing_labels(item.get("filing_ids")),
                }
            )

    if active_view not in {"ingestion", "query", "history"}:
        active_view = "ingestion"

    mode_labels = {
        "decomposer": "Query Decomposer (RAG)",
        "react": "ReAct Agent",
        "reflexion": "Reflexion Agent",
    }

    return render_template_string(
        BASE_TEMPLATE,
        filings=filings,
        history_preview=history_preview,
        history_full=history_full,
        query_result=query_result,
        query_mode=query_mode,
        query_mode_label=mode_labels.get(query_mode, query_mode),
        query_text=query_text,
        selected_filings=selected_filings or [],
        active_view=active_view,
    )


def _run_rag_query(question: str, filing_ids: Optional[List[int]]) -> str:
    rag_chain = RAGChain(app_config, vector_store, table_retrieval_func=persistent_store.get_table_by_id)
    print("obj initialised")
    return rag_chain.query(question, ids_to_retrieve=filing_ids)


def _run_react_query(question: str, filing_ids: Optional[List[int]]) -> str:
    director_strings = persistent_store.get_director_snippets()
    agent = ReactAgent(app_config, director_strings, persistent_store)
    agent.initialize_agent(vector_store, ids_to_retrieve=filing_ids)
    return agent.run(question)


def _run_reflexion_query(question: str, filing_ids: Optional[List[int]]) -> str:
    director_strings = persistent_store.get_director_snippets()
    agent = ReactWithReflection(app_config, director_strings, persistent_store)
    agent.initialize_agent(vector_store, ids_to_retrieve=filing_ids)
    return agent.run(question)


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    view = request.args.get("view", "ingestion")
    if view not in {"ingestion", "query", "history"}:
        view = "ingestion"
    return _render_dashboard(active_view=view)


@app.route("/add-filing", methods=["POST"])
def add_filing():
    name = request.form.get("filing_name", "").strip()
    urls_raw = request.form.get("filing_urls", "")
    urls = [line.strip() for line in urls_raw.splitlines() if line.strip()]

    if not name:
        flash("Please provide a filing name.")
        return redirect(url_for("index", view="ingestion"))

    if not urls:
        flash("Please provide at least one URL.")
        return redirect(url_for("index", view="ingestion"))

    try:
        entry = persistent_store.add_filing(name, urls)
        filing_id = entry["id"]

        processor = TextPreProcessor(app_config, [], filing_id)
        documents, director_snippets = processor.load_and_process_filings(
            [(name, url) for url in urls],
            filing_id,
        )

        print("document loaded")

        if documents:
            vector_store.add_chunk_documents(documents)

        if director_snippets:
            persistent_store.save_director_snippets(filing_id, director_snippets)

        flash(f"Filing '{name}' ingested with ID {filing_id}.")
    except Exception as exc:
        logger.exception("Failed to ingest filing")
        flash(f"Failed to ingest filing: {exc}")

    return redirect(url_for("index", view="ingestion"))


@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    filing_id_raw = request.form.get("filing_id", "")
    file = request.files.get("pdf_file")

    if not filing_id_raw:
        flash("Please choose a filing before uploading.")
        return redirect(url_for("index", view="ingestion"))

    try:
        filing_id = int(filing_id_raw)
    except ValueError:
        flash("Invalid filing ID provided.")
        return redirect(url_for("index", view="ingestion"))

    if not file or file.filename == "":
        flash("No PDF selected.")
        return redirect(url_for("index", view="ingestion"))

    filename = secure_filename(file.filename)
    stored_name = f"{filing_id}_{filename}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    file.save(file_path)

    try:
        processor = TablePreProcessor(app_config)
        table_docs, table_records = processor.process_document(file_path, filing_id)

        if table_docs:
            vector_store.add_table_documents(table_docs)
        if table_records:
            persistent_store.save_tables(table_records)

        flash(f"Extracted {len(table_docs)} table summaries for filing ID {filing_id}.")
    except Exception as exc:
        logger.exception("Failed to extract tables")
        flash(f"Failed to extract tables: {exc}")

    return redirect(url_for("index", view="ingestion"))


@app.route("/query", methods=["POST"])
def run_query():
    question = request.form.get("query_text", "").strip()
    mode = request.form.get("query_mode", "decomposer")
    filing_ids_raw = request.form.getlist("filing_ids")

    if not question:
        flash("Please provide a question to run.")
        return redirect(url_for("index", view="query"))

    filing_ids: Optional[List[int]] = None
    if filing_ids_raw:
        try:
            filing_ids = [int(fid) for fid in filing_ids_raw]
        except ValueError:
            flash("Invalid filing selection.")
            return redirect(url_for("index", view="query"))

    print(mode)
    print(question)
    print(filing_ids)

    try:
        if mode == "react":
            answer = _run_react_query(question, filing_ids)
        elif mode == "reflexion":
            answer = _run_reflexion_query(question, filing_ids)
        else:
            answer = _run_rag_query(question, filing_ids)
            mode = "decomposer"

        persistent_store.add_query_history(question, answer, mode, filing_ids or [])

        return _render_dashboard(
            query_result=answer,
            query_mode=mode,
            query_text=question,
            selected_filings=filing_ids or [],
            active_view="query",
        )
    except Exception as exc:
        logger.exception("Query failed")
        flash(f"Query failed: {exc}")
        return redirect(url_for("index", view="query"))


@app.route("/history", methods=["GET"])
def query_history():
    return redirect(url_for("index", view="history"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
