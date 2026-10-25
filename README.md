# llm-reliability-kit

Pattern-uri de producție pentru apeluri către LLM-uri, scrise de mână, fără framework:
retry cu backoff, fallback între provideri, structured outputs validate cu Pydantic,
buclă de tool calling, cost per conversație.

> Lucru în desfășurare — Pasul 2 din planul meu de studiu (LLM APIs).

```bash
pip install -r requirements.txt
pytest -q
```
