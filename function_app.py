import logging
import os
import openai
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import azure.functions as func

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="ragquery", methods=["GET"])
def ragquery(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("RAG API -kutsu vastaanotettu.")

    question = req.params.get("q")
    if not question:
        return func.HttpResponse("Anna parametri ?q= kysymykselle.", status_code=400)

    try:
        # 🔹 Lue ympäristömuuttujat
        search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        search_index = os.getenv("AZURE_SEARCH_INDEX")
        search_key = os.getenv("AZURE_SEARCH_KEY")
        openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        openai_key = os.getenv("AZURE_OPENAI_KEY")

        # 1️⃣ Hae dokumentit Azure Cognitive Searchista
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=search_index,
            credential=AzureKeyCredential(search_key)
        )

        results = list(search_client.search(search_text=question, top=3))
        logging.info(f"Hakutuloksia: {len(results)}")

        if not results:
            return func.HttpResponse("En tiedä, koska hakutuloksia ei löytynyt.", status_code=200)

        # 2️⃣ Luo konteksti ja lähteet
        context = "\n".join([doc.get("chunk", "") for doc in results if doc.get("chunk")])
        sources = [f"- {doc.get('title', 'tuntematon.pdf')}" for doc in results]

        if not context.strip():
            return func.HttpResponse(
                f"En tiedä, koska kontekstista ei löytynyt relevanttia tietoa.\n\n**Lähteet:**\n" + "\n".join(sources),
                status_code=200
            )

        # 3️⃣ Luo prompt
        prompt = (
            f"Vastaa kysymykseen käyttäen seuraavaa kontekstia:\n{context}\n\n"
            f"Kysymys: {question}\n\n"
            "Tiivistä vastaus enintään 3 lauseeseen. Jos konteksti ei sisällä vastausta, sano 'En tiedä'."
        )

        # 4️⃣ Kutsu Azure OpenAI
        response = openai.ChatCompletion.create(
            api_base=openai_endpoint,
            api_key=openai_key,
            api_type="azure",
            api_version="2023-09-01-preview",
            engine=openai_deployment,
            messages=[
                {"role": "system", "content": "Olet avulias assistentti. Vastaa tiiviisti, enintään 3 lauseella."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )

        answer = response["choices"][0]["message"]["content"].strip()

        final_answer = f"{answer}\n\n**Lähteet:**\n" + "\n".join(sources)
        return func.HttpResponse(final_answer, status_code=200)

    except Exception as e:
        logging.exception("Virhe RAG-kyselyssä")
        return func.HttpResponse(f"Virhe: {str(e)}", status_code=500)
