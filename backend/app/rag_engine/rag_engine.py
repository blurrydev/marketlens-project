import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


class MarketLensRAG:

    def __init__(self, user_api_key : str, temp=0.0):
        self.api_key = user_api_key

        if not self.api_key:
            raise ValueError(
                "CRITICAL ERROR: GOOGLE_API_KEY is missing."
            )

        print("Initializing LLM for MarketLens...")

        self.llm = ChatGoogleGenerativeAI(
                model = "gemini-2.5-flash-image", 
                temperature = temp,
                api_key = self.api_key
        )


    def generate_answer(self, query: str, retrived_contexts: List[str]) -> str:

        if not retrived_contexts:
            return "I cannot find the answer in the provided documents"

        print(f"Stitching together {len(retrived_contexts)} context chunks.")

        context_string = "\n\n---\n\n".join(retrived_contexts)

        final_prompt = f"""
                            You are MarketLens, an expert financial AI assistant. 
                            Read the following retrieved context from corporate financial documents. 
                            Use ONLY this context to answer the user's question. 
                            If the answer is not in the context, explicitly say "I cannot find the answer in the provided documents."
                            Do not make up numbers or hallucinate.

                            Context:
                            {context_string}

                            User Question: {query}
                        """

        print("Sending prompt to Gemini")


        try:

            response = self.llm.invoke([HumanMessage(content=final_prompt)])
            return response.content

        except Exception as e:
            print(f"ERROR: Error while invoking Gemini: {e}")
            return "Sorry, I encountered an error while generating response"
