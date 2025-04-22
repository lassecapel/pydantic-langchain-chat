from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Laad OpenAI API-key uit .env
load_dotenv()

# Definieer gestructureerde output voor de assistent
class FinanceAdvice(BaseModel):
    antwoord: str = Field(description="Het antwoord op de financiële vraag")
    actie: str = Field(description="Eventuele actie die de medewerker moet ondernemen, of 'geen'")

# System prompt: rol van de agent
system_prompt = SystemMessage(
    "Je bent een behulpzame financiële assistent voor medewerkers van een accountantskantoor. "
    "Beantwoord vragen kort, duidelijk en praktisch. Geef indien nodig een actie aan ('geen' als er geen actie is)."
)

# Initialiseer OpenAI chatmodel
llm = ChatOpenAI(model="gpt-4")

# Start CLI-chat
def main():
    print("Welkom bij de Financiële Assistent. Typ 'exit' om af te sluiten.")
    conversation = [system_prompt]

    while True:
        user_input = input("Medewerker: ")
        if user_input.lower() == "exit":
            print("Chat afgesloten.")
            break

        # Voeg menselijke input toe aan conversatie
        human_msg = HumanMessage(user_input)
        conversation.append(human_msg)

        # Maak prompt template voor gestructureerde output
        prompt = ChatPromptTemplate.from_messages(conversation)

        # Vraag het model om gestructureerd antwoord
        # (Langchain: forceren van gestructureerde output vereist extra setup,
        # hier simuleren we het door het model te vragen in JSON te antwoorden)
        prompt_text = (
            f"Beantwoord als JSON met de velden 'antwoord' en 'actie'. Vraag: {user_input}"
        )
        response = llm.invoke([system_prompt, HumanMessage(prompt_text)])
        print("Assistent:", response.content)

        # Voeg AI-antwoord toe aan conversatie
        conversation.append(AIMessage(response.content))

if __name__ == "__main__":
    main()
