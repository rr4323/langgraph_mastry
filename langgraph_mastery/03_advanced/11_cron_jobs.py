"""
This script demonstrates how to schedule and manage cron jobs on the LangGraph Platform.

Cron jobs allow you to run an assistant on a user-defined schedule. This is useful for
tasks like sending daily reports, performing regular data analysis, or any other automated,
recurring workflow.

This script is a client-side example that uses the `langgraph_sdk` to interact with a
deployed assistant on the LangGraph Platform. It covers:
- Creating a stateless cron job that runs on a schedule with a fresh state every time.
- Creating a stateful cron job that runs on a schedule within a specific, persistent thread.
- Listing all active cron jobs for an assistant.
- Deleting cron jobs to clean up and avoid unintended resource usage.

Prerequisites:
1.  **A Deployed Assistant**: You must have a LangGraph assistant deployed on the LangGraph
    Platform. You can deploy an assistant using the `langgraph` CLI.
2.  **API Key**: You need an API key for your LangGraph Platform account.
3.  **`langgraph_sdk`**: The Python SDK must be installed (`pip install langgraph_sdk`).

To run this script:
1.  Set the following environment variables:
    - `LANGGRAPH_API_KEY`: Your API key for the LangGraph Platform.
    - `ASSISTANT_ID`: The ID of your deployed assistant.
2.  Run the script from your terminal:
    ```bash
    python langgraph_mastery/03_advanced/11_cron_jobs.py
    ```

Note: The script uses a placeholder cron schedule (`*/5 * * * *` - every 5 minutes) for
demonstration. In a real application, you would adjust this to your desired frequency.
It's crucial to delete cron jobs when they are no longer needed to prevent unwanted API usage.
"""

import asyncio
import os
import uuid

from langgraph_sdk import LangGraphClient


async def main():
    """Main function to demonstrate creating, listing, and deleting cron jobs."""
    # --- 1. Setup --- 
    # Initialize the LangGraph client. It will automatically use the
    # LANGGRAPH_API_KEY from your environment variables.
    client = LangGraphClient()

    # Get the assistant ID from environment variables.
    assistant_id = os.environ.get("ASSISTANT_ID")
    if not assistant_id:
        raise ValueError("The ASSISTANT_ID environment variable is not set.")

    print(f"Using Assistant ID: {assistant_id}")

    # --- 2. Create a Stateless Cron Job --- 
    # This job runs on the specified schedule, creating a new, stateless run each time.
    print("\n--- Creating a stateless cron job... ---")
    # Cron schedule: every 5 minutes for demonstration purposes.
    # See https://crontab.guru/ for help defining schedules.
    stateless_cron_schedule = "*/5 * * * *"
    stateless_cron_input = {"messages": [{"role": "user", "content": "Generate a daily news summary."}]}

    try:
        stateless_cron_job = await client.crons.create(
            assistant_id,
            schedule=stateless_cron_schedule,
            input=stateless_cron_input,
        )
        print(f"Successfully created stateless cron job with ID: {stateless_cron_job['cron_id']}")
        print(f"It will run every 5 minutes.")
    except Exception as e:
        print(f"Error creating stateless cron job: {e}")
        stateless_cron_job = None

    # --- 3. Create a Stateful Cron Job (on a Thread) --- 
    # First, create a new thread for the stateful job.
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    print(f"\n--- Creating a stateful cron job on thread {thread_id}... ---")

    # This job runs on the same schedule but within the context of the created thread.
    # This means the conversation history is maintained between runs.
    stateful_cron_schedule = "*/5 * * * *"
    stateful_cron_input = {"messages": [{"role": "user", "content": "What time is it? Check if I have any new alerts."}]}

    try:
        stateful_cron_job = await client.crons.create_for_thread(
            thread_id,
            assistant_id,
            schedule=stateful_cron_schedule,
            input=stateful_cron_input,
        )
        print(f"Successfully created stateful cron job with ID: {stateful_cron_job['cron_id']}")
        print(f"It will run every 5 minutes on thread {thread_id}.")
    except Exception as e:
        print(f"Error creating stateful cron job: {e}")
        stateful_cron_job = None

    # --- 4. List Cron Jobs --- 
    print("\n--- Listing all active cron jobs for the assistant... ---")
    try:
        active_crons = await client.crons.list(assistant_id=assistant_id)
        if active_crons:
            for cron in active_crons:
                print(f"- Cron ID: {cron['cron_id']}, Schedule: '{cron['schedule']}', Type: {'Stateful (Thread)' if cron.get('thread_id') else 'Stateless'}")
        else:
            print("No active cron jobs found.")
    except Exception as e:
        print(f"Error listing cron jobs: {e}")

    # --- 5. Cleanup --- 
    # It is very important to delete cron jobs that are no longer needed!
    print("\n--- Cleaning up created cron jobs... ---")
    if stateless_cron_job:
        try:
            await client.crons.delete(stateless_cron_job["cron_id"])
            print(f"Successfully deleted stateless cron job: {stateless_cron_job['cron_id']}")
        except Exception as e:
            print(f"Error deleting stateless cron job {stateless_cron_job['cron_id']}: {e}")

    if stateful_cron_job:
        try:
            await client.crons.delete(stateful_cron_job["cron_id"])
            print(f"Successfully deleted stateful cron job: {stateful_cron_job['cron_id']}")
        except Exception as e:
            print(f"Error deleting stateful cron job {stateful_cron_job['cron_id']}: {e}")


if __name__ == "__main__":
    # Ensure you have `langgraph_sdk` installed: pip install langgraph_sdk
    # Set LANGGRAPH_API_KEY and ASSISTANT_ID in your environment.
    try:
        asyncio.run(main())
    except ValueError as e:
        print(f"Error: {e}")
