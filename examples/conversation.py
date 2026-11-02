"""Two-turn conversation stored in Pixeltable.

Runs offline with a scripted model. No API key.
"""

from __future__ import annotations

import asyncio

from agents import Agent, RunConfig, Runner
from agents.testing import ScriptedModel, assistant_message

from openai_agents_pixeltable import PixeltableSession


async def main() -> str:
    session = PixeltableSession("demo", table_name="examples.conversation")
    await session.clear_session()
    model = ScriptedModel(
        [
            [assistant_message("San Francisco.")],
            [assistant_message("California.")],
        ]
    )
    agent = Agent(name="Assistant", instructions="Reply concisely.", model=model)
    config = RunConfig(tracing_disabled=True)
    await Runner.run(agent, "What city is the Golden Gate Bridge in?", session=session, run_config=config)
    result = await Runner.run(agent, "What state is it in?", session=session, run_config=config)
    return str(result.final_output)


if __name__ == "__main__":
    print(asyncio.run(main()))
