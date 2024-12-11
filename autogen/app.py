from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
import asyncio
from autogen_agentchat.ui import Console
from tavily import TavilyClient

def get_sent_emails():
  mock_emails = [
        {
            "recipient": "john.doe@siemens.de",
            "subject": "Anfrage Wachmaschiene",
            "content": "Ich hattte eine Wachmaschine gekauft und wollte wissen, wann ich sie erhalten kann.",
            "timestamp": "2024-12-11 09:30:00",
            "status": "delivered"
        },
        {
            "recipient": "bob.ross@ticketsbolivia.com",
            "subject": "Siehe Anhang",
            "content": "Wie besprochen anbei die gewünschten Informationen.",
            "timestamp": "2024-12-11 14:15:00",
            "status": "delivered"
        },
    ]
    
  return mock_emails

def search_web(query: str) -> str:
    tavily_client = TavilyClient(api_key="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")
    response = tavily_client.search(query)
    # Return the first result's content or a default message
    if response and response.get('results'):
        return response['results'][0].get('content', 'No results found.')
    return 'No results found.'

def save_to_file(content: str, filename: str = "research_summary.txt") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


async def main() -> None:
  summary_agent = AssistantAgent(
      name="summaryAgent", 
      system_message="""
        Du bist ein Agent der die Rechercheergebnisse über GPUs und Hardware zusammenfasst.
        Erstelle eine strukturierte Zusammenfassung mit folgenden Punkten:

        - Aktuelle GPU Modelle und deren Preise
        - Verfügbarkeit der Modelle
        - Preis-Leistungs-Empfehlungen
        - Technische Details (VRAM, Rechenleistung, Energieeffizienz)
        
        Speichere am Ende die Zusammenfassung in eine Textdatei.
      """,
      model_client=OpenAIChatCompletionClient(
              model="gpt-4o-2024-08-06",
          ),
      tools=[save_to_file]
  )

  research_agent = AssistantAgent(
      name="researchAgent",
      system_message="""
        Du bist ein Experte für LLM Hardware und GPUs.
        Recherchiere aktuelle Preise und Verfügbarkeiten von GPUs die für LLM Training und Inferenz geeignet sind.
        Analysiere das Preis-Leistungs-Verhältnis und gib Empfehlungen für verschiedene Anwendungsfälle.
        Berücksichtige dabei Faktoren wie VRAM, Rechenleistung und Energieeffizienz.
      """,
      model_client=OpenAIChatCompletionClient(
              model="gpt-4o-2024-08-06",
          ),
      tools=[search_web]
  )

    # Define termination condition
  termination = TextMentionTermination("TERMINATE")

  agent_team = RoundRobinGroupChat([research_agent, summary_agent], termination_condition=termination)

  # Run the team and stream messages to the console
  stream = agent_team.run_stream(task="Recheriere bitte eine Lösung mit der wir on-premise eine Llama 70B laufen lassen könnnen und trainieren könnnen.")
  await Console(stream)

asyncio.run(main())