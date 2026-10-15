import asyncio
import os

from openai import AsyncOpenAI


async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:4000/v1",
        api_key=os.environ["LITELLM_MASTER_KEY"],
    )

    try:
        response = await client.chat.completions.create(
            model="fast-llm",
            messages=[
                {
                    "role": "user",
                    "content": "Explain AI agent architecture in three sentences.",
                }
            ],
        )

        print("Gemini response:")
        print(response.choices[0].message.content)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())