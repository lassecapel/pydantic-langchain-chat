# Import future annotations to allow for type annotations that reference the class being defined
from __future__ import annotations as _annotations

# Standard library imports
import asyncio  # For asynchronous programming
import json  # For JSON serialization/deserialization
import sqlite3  # For database operations
from collections.abc import AsyncIterator  # For typing async iterators
from concurrent.futures.thread import ThreadPoolExecutor  # For running blocking code in threads
from contextlib import asynccontextmanager  # For creating async context managers
from dataclasses import dataclass  # For creating data classes
from datetime import datetime, timezone  # For timestamp handling
from functools import partial  # For creating partial functions
from pathlib import Path  # For file path handling
from typing import Annotated, Any, Callable, Literal, TypeVar  # For type annotations
from dotenv import load_dotenv

# Third-party imports
import fastapi  # Web framework for building APIs
import logfire  # For logging and instrumentation
from fastapi import Depends, Request  # FastAPI dependencies
from fastapi.responses import FileResponse, Response, StreamingResponse  # FastAPI response types
from typing_extensions import LiteralString, ParamSpec, TypedDict  # Extended typing utilities

# Pydantic AI imports - this is the core library for AI agent functionality
from pydantic_ai import Agent  # Main agent class for AI interactions
from pydantic_ai.exceptions import UnexpectedModelBehavior  # Exception for unexpected model behavior
from pydantic_ai.messages import (  # Message types for AI interactions
    ModelMessage,  # Base message class
    ModelMessagesTypeAdapter,  # For validating message JSON
    ModelRequest,  # User request message
    ModelResponse,  # AI response message
    TextPart,  # Text content part
    UserPromptPart,  # User prompt part
)

load_dotenv()

# Comment out logfire configuration to avoid dependency issues
# logfire.configure(send_to_logfire='if-token-present')

# Initialize the AI agent with OpenAI's GPT-4o model
# The instrument parameter is set to False to avoid logfire dependency issues
agent = Agent('openai:gpt-4o', instrument=False)

# Get the directory where this script is located
THIS_DIR = Path(__file__).parent

# Define the application lifespan context manager
# This manages the lifecycle of the database connection
@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    # Connect to the database when the application starts
    async with Database.connect() as db:
        # Yield the database connection to the application
        yield {'db': db}
    # Database connection is automatically closed when the application shuts down

# Initialize the FastAPI application with the lifespan context manager
app = fastapi.FastAPI(lifespan=lifespan)

# Comment out logfire instrumentation to avoid dependency issues
# logfire.instrument_fastapi(app)

# Define route for the root URL - serves the HTML chat interface
@app.get('/')
async def index() -> FileResponse:
    # Return the HTML file for the chat application
    return FileResponse((THIS_DIR / 'chat_app.html'), media_type='text/html')

# Define route for the TypeScript code - serves the TS file that powers the frontend
@app.get('/chat_app.ts')
async def main_ts() -> FileResponse:
    """Get the raw typescript code, it's compiled in the browser, forgive me."""
    # Return the TypeScript file as plain text - it will be compiled in the browser
    return FileResponse((THIS_DIR / 'chat_app.ts'), media_type='text/plain')

# Dependency function to get the database connection from the request state
async def get_db(request: Request) -> Database:
    # Extract the database connection that was set up in the lifespan context manager
    return request.state.db

# Define route to get chat history
@app.get('/chat/')
async def get_chat(database: Database = Depends(get_db)) -> Response:
    # Retrieve all messages from the database
    msgs = await database.get_messages()
    
    # Convert each message to the chat message format and join them with newlines
    # Each message is JSON-encoded and UTF-8 encoded
    return Response(
        b'\n'.join(json.dumps(to_chat_message(m)).encode('utf-8') for m in msgs),
        media_type='text/plain',
    )

# Define the structure of chat messages sent to the browser
class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal['user', 'model']  # Who sent the message: 'user' or 'model' (AI)
    timestamp: str  # When the message was sent (ISO format)
    content: str  # The actual message content

# Convert a ModelMessage to a ChatMessage for the frontend
def to_chat_message(m: ModelMessage) -> ChatMessage:
    # Get the first part of the message (messages can have multiple parts)
    first_part = m.parts[0]
    
    # If the message is a user request
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            # Ensure the content is a string
            assert isinstance(first_part.content, str)
            # Return a user message
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    # If the message is a model (AI) response
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            # Return a model message
            return {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': first_part.content,
            }
    
    # If the message doesn't match expected formats, raise an exception
    raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')

# Define route to send a new message to the chat
@app.post('/chat/')
async def post_chat(
    # Get the user's prompt from the form data
    prompt: Annotated[str, fastapi.Form()], 
    # Get the database connection
    database: Database = Depends(get_db)
) -> StreamingResponse:
    # Define an async generator function to stream messages back to the client
    async def stream_messages():
        """Streams new line delimited JSON `Message`s to the client."""
        # First, immediately stream the user prompt back so it can be displayed right away
        yield (
            json.dumps(
                {
                    'role': 'user',
                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                    'content': prompt,
                }
            ).encode('utf-8')
            + b'\n'
        )
        
        # Get the chat history so far to provide context to the AI agent
        messages = await database.get_messages()
        
        # Run the agent with the user prompt and the chat history
        # Using a streaming context to get responses as they're generated
        async with agent.run_stream(prompt, message_history=messages) as result:
            # Stream each piece of text as it's generated
            # debounce_by=0.01 means we'll get updates roughly every 10ms
            async for text in result.stream(debounce_by=0.01):
                # Convert the raw text to a ModelResponse with a TextPart
                m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                # Convert to a ChatMessage, JSON-encode it, and yield it with a newline
                yield json.dumps(to_chat_message(m)).encode('utf-8') + b'\n'

        # After the streaming is complete, save all new messages to the database
        # This includes both the user prompt and the complete AI response
        await database.add_messages(result.new_messages_json())

    # Return a streaming response that will call our generator function
    return StreamingResponse(stream_messages(), media_type='text/plain')

# Type parameters for generic function types
P = ParamSpec('P')  # For function parameters
R = TypeVar('R')    # For function return types

@dataclass
class Database:
    """Rudimentary database to store chat messages in SQLite.

    The SQLite standard library package is synchronous, so we
    use a thread pool executor to run queries asynchronously.
    This allows us to use SQLite with FastAPI's async framework.
    """

    con: sqlite3.Connection         # SQLite connection
    _loop: asyncio.AbstractEventLoop  # Event loop for async operations
    _executor: ThreadPoolExecutor   # Thread pool for running SQLite operations

    # Class method to create a database connection as an async context manager
    @classmethod
    @asynccontextmanager
    async def connect(
        cls, file: Path = THIS_DIR / '.chat_app_messages.sqlite'  # Default database file location
    ) -> AsyncIterator[Database]:
        # Removed logfire span to avoid dependency issues
        # Get the current event loop
        loop = asyncio.get_event_loop()
        # Create a thread pool with a single worker (SQLite is single-threaded)
        executor = ThreadPoolExecutor(max_workers=1)
        # Run the _connect method in the thread pool to avoid blocking the event loop
        con = await loop.run_in_executor(executor, cls._connect, file)
        # Create a new Database instance with the connection, loop, and executor
        slf = cls(con, loop, executor)
        try:
            # Yield the database instance to the caller
            yield slf
        finally:
            # Ensure the connection is closed when the context manager exits
            await slf._asyncify(con.close)

    # Static method to create a SQLite connection and set up the database
    @staticmethod
    def _connect(file: Path) -> sqlite3.Connection:
        # Connect to the SQLite database file
        con = sqlite3.connect(str(file))
        # Comment out logfire instrumentation to avoid dependency issues
        # con = logfire.instrument_sqlite3(con)
        # Create a cursor for executing SQL commands
        cur = con.cursor()
        # Create the messages table if it doesn't exist
        # - id: auto-incrementing primary key
        # - message_list: JSON text containing the message data
        cur.execute(
            'CREATE TABLE IF NOT EXISTS messages (id INT PRIMARY KEY, message_list TEXT);'
        )
        # Commit the changes
        con.commit()
        # Return the connection
        return con

    # Method to add new messages to the database
    async def add_messages(self, messages: bytes):
        # Execute an INSERT query to add the messages to the database
        # The messages are passed as a byte string containing JSON data
        await self._asyncify(
            self._execute,
            'INSERT INTO messages (message_list) VALUES (?);',
            messages,
            commit=True,
        )
        # Commit the transaction to save the changes
        await self._asyncify(self.con.commit)

    # Method to retrieve all messages from the database
    async def get_messages(self) -> list[ModelMessage]:
        # Execute a SELECT query to get all messages, ordered by ID (chronological order)
        c = await self._asyncify(
            self._execute, 'SELECT message_list FROM messages order by id'
        )
        # Fetch all the rows from the query result
        rows = await self._asyncify(c.fetchall)
        # Initialize an empty list to store the messages
        messages: list[ModelMessage] = []
        # For each row, parse the JSON message data and add it to the list
        for row in rows:
            # Use the ModelMessagesTypeAdapter to validate and convert the JSON to ModelMessage objects
            # This extends the messages list with all messages from this row
            messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
        # Return the complete list of messages
        return messages

    # Helper method to execute SQL queries
    def _execute(
        self, sql: LiteralString, *args: Any, commit: bool = False
    ) -> sqlite3.Cursor:
        # Create a cursor for executing the query
        cur = self.con.cursor()
        # Execute the SQL query with the provided arguments
        cur.execute(sql, args)
        # If commit is True, commit the changes to the database
        if commit:
            self.con.commit()
        # Return the cursor for fetching results
        return cur

    # Helper method to run synchronous functions asynchronously
    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        # Run the function in the thread pool executor to avoid blocking the event loop
        # This allows us to use synchronous SQLite functions in an async context
        return await self._loop.run_in_executor(  # type: ignore
            self._executor,
            # Create a partial function with the keyword arguments
            partial(func, **kwargs),
            # Pass the positional arguments
            *args,  # type: ignore
        )

# Entry point for running the application directly
if __name__ == '__main__':
    # Import uvicorn for running the FastAPI application
    import uvicorn

    # Run the FastAPI application with uvicorn
    # - reload=True: Enable auto-reload when files change (development mode)
    # - reload_dirs: Directories to watch for changes
    uvicorn.run(
        'chat:app', reload=True, reload_dirs=[str(THIS_DIR)]
    )