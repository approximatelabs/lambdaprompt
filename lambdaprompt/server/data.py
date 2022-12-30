from aiosqlite import Connection as Database


async def setup_database(db: Database):
    query = f"""
        CREATE TABLE IF NOT EXISTS backgroundTasks (
            backgroundTaskId TEXT,
            logEventId TEXT
        );
    """
    await db.execute(query)
    query = f"""
        CREATE TABLE IF NOT EXISTS logEvents (
            logEventId TEXT PRIMARY KEY,
            logAsJson TEXT
        );
    """
    await db.execute(query)
    query = f"""
        CREATE TABLE IF NOT EXISTS prompts (
            promptid TEXT PRIMARY KEY,
            prompt TEXT
        );
    """
    await db.execute(query)


async def add_prompt_if_not_exists(db: Database, prompt_id: str, prompt: str):
    query = f"""
        INSERT OR IGNORE INTO prompts (promptId, prompt)
        VALUES (:promptId, :prompt);
    """
    await db.execute(query, parameters={"promptId": prompt_id, "prompt": prompt})


async def log_event(
    db: Database, log_event_id: str, log_as_json: str, background_task_id: str = None
):
    query = f"""
        INSERT INTO logEvents (logEventId, logAsJson)
        VALUES (:logEventId, :logAsJson);
    """
    await db.execute(
        query, parameters={"logEventId": log_event_id, "logAsJson": log_as_json}
    )

    if background_task_id:
        query = f"""
            INSERT INTO backgroundTasks (backgroundTaskId, logEventId)
            VALUES (:backgroundTaskId, :logEventId);
        """
        await db.execute(
            query,
            parameters={
                "backgroundTaskId": background_task_id,
                "logEventId": log_event_id,
            },
        )


async def get_logs_for_task(db: Database, background_task_id: str):
    query = f"""
        SELECT logAsJson FROM logEvents
        INNER JOIN backgroundTasks
        ON logEvents.logEventId = backgroundTasks.logEventId
        WHERE backgroundTasks.backgroundTaskId = :backgroundTaskId;
    """
    return [
        l
        for l, in await db.execute_fetchall(
            query, parameters={"backgroundTaskId": background_task_id}
        )
    ]
