#!/usr/bin/env python3
"""
RAG API Server with document processing and transcription capabilities
"""

import os
import uuid
import json
import shutil
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

import aiofiles
import ffmpeg
import pytube
import whisper
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG API Server")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed_documents")
DB_DIR = Path("chroma_db")
TEMP_DIR = Path("temp")

for directory in [UPLOAD_DIR, PROCESSED_DIR, DB_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# Models for request/response data
class DocumentResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    size: int
    upload_time: str
    status: str
    document_type: str

class YouTubeResponse(BaseModel):
    id: str
    video_id: str
    title: str
    upload_time: str
    status: str
    duration: Optional[int] = None
    thumbnail: Optional[str] = None

class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str

class VideoResponse(BaseModel):
    id: str
    filename: str
    upload_time: str
    status: str
    duration: Optional[int] = None
    segments: Optional[List[TranscriptSegment]] = None

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str

class YouTubeRequest(BaseModel):
    url: HttpUrl

class DocumentMetadata(BaseModel):
    id: str
    filename: str
    content_type: str
    size: int
    upload_time: str
    document_type: str
    status: str
    segments: Optional[List[TranscriptSegment]] = None
    youtube_id: Optional[str] = None
    youtube_title: Optional[str] = None
    youtube_thumbnail: Optional[str] = None
    duration: Optional[int] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket) -> str:
        """Accept a WebSocket connection and assign a unique ID"""
        await websocket.accept()
        connection_id = str(self.connection_count)
        self.active_connections[connection_id] = websocket
        self.connection_count += 1
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
    
    async def send_json(self, connection_id: str, data: Dict):
        """Send a JSON message to a specific connection"""
        if connection_id in self.active_connections:
            await self.active_connections[connection_id].send_json(data)

manager = ConnectionManager()

# Streaming callback handler
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, connection_id: str):
        self.connection_id = connection_id
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Stream tokens as they're generated"""
        await manager.send_json(self.connection_id, {
            "type": "stream",
            "content": token
        })

# RAG Chat Application with document processing
class RAGProcessor:
    def __init__(self, db_dir: str = "chroma_db"):
        self.db_dir = db_dir
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.conversation_chains: Dict[str, Any] = {}
        self.memories: Dict[str, ConversationBufferMemory] = {}
        self.document_metadata: Dict[str, DocumentMetadata] = {}
        
        # Load existing document metadata if available
        metadata_path = Path("document_metadata.json")
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    self.document_metadata = {k: DocumentMetadata(**v) for k, v in json.load(f).items()}
            except Exception as e:
                print(f"Error loading document metadata: {e}")
        
        # Initialize whisper model for transcription
        self.whisper_model = whisper.load_model("base")
    
    def save_metadata(self):
        """Save document metadata to disk"""
        with open("document_metadata.json", "w") as f:
            json.dump({k: v.dict() for k, v in self.document_metadata.items()}, f)
    
    async def broadcast_document_status(self, document_id: str):
        """Broadcast document status update to all connected clients"""
        if document_id not in self.document_metadata:
            return
            
        document = self.document_metadata[document_id]
        # Send update to all connected clients
        for connection_id in manager.active_connections:
            try:
                await manager.send_json(connection_id, {
                    "type": "document_update",
                    "document": document.dict()
                })
            except Exception as e:
                print(f"Error sending document update to {connection_id}: {e}")
    
    async def initialize_for_connection(self, connection_id: str, streaming_handler: Optional[BaseCallbackHandler] = None):
        """Initialize the memory and conversation chain for a connection"""
        # Create memory for this connection
        self.memories[connection_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Check if vector store exists
        if not self.vectorstore:
            print("Vector store not initialized. Loading documents...")
            await self.load_documents()
            
        if not self.vectorstore:
            print("Failed to initialize vector store")
            return False
        
        # Create LLM with streaming
        llm = ChatOpenAI(
            temperature=0,
            streaming=True,
            callbacks=[streaming_handler] if streaming_handler else None
        )
        
        # Create conversation chain for this connection
        custom_template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say you don't know. Don't try to make up an answer.
        
        {context}
        
        Chat History: {chat_history}
        Question: {question}
        
        Important: Provide only your answer without repeating the question or including phrases like "Based on the context" or "According to the documents". Start your response directly with the relevant information.
        
        Answer:
        """
        
        CUSTOM_PROMPT = PromptTemplate.from_template(custom_template)
        
        self.conversation_chains[connection_id] = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=self.memories[connection_id],
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
        )
        
        return True
    
    def cleanup_connection(self, connection_id: str):
        """Clean up resources for a connection"""
        if connection_id in self.memories:
            del self.memories[connection_id]
        
        if connection_id in self.conversation_chains:
            del self.conversation_chains[connection_id]
    
    async def load_documents(self):
        """Load processed documents and create vector store"""
        try:
            # Check if processed documents exist
            if not os.path.exists(PROCESSED_DIR) or not any(os.listdir(PROCESSED_DIR)):
                print(f"No processed documents found in {PROCESSED_DIR}")
                return False
                
            # Load all text documents from processed directory
            loader = DirectoryLoader(
                PROCESSED_DIR,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            
            if not documents:
                print(f"No documents loaded from {PROCESSED_DIR}")
                return False
                
            print(f"Loaded {len(documents)} documents")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            print(f"Split into {len(splits)} chunks")
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.db_dir
            )
            
            return True
        except Exception as e:
            print(f"Error loading documents: {e}")
            return False
    
    async def chat(self, connection_id: str, query: str) -> str:
        """
        Process a chat query using the conversation chain for a specific connection
        
        Args:
            connection_id: The WebSocket connection ID
            query: User's question or message
            
        Returns:
            Response from the model
        """
        if connection_id not in self.conversation_chains:
            return "Connection not initialized. Please reconnect."
            
        if not self.vectorstore:
            return "No documents have been processed yet."
        
        try:
            response = await self.conversation_chains[connection_id].ainvoke({"question": query})
            return response["answer"]
        except Exception as e:
            print(f"Error in chat: {e}")
            return f"An error occurred: {str(e)}"
    
    def clear_memory(self, connection_id: str):
        """Clear the conversation memory for a specific connection"""
        if connection_id in self.memories:
            self.memories[connection_id].clear()
            return True
        return False
    
    async def process_document(self, file_id: str, file_path: str, original_filename: str, content_type: str):
        """Process an uploaded document based on its type"""
        try:
            # Set status to processing
            self.document_metadata[file_id].status = "processing"
            self.save_metadata()
            await self.broadcast_document_status(file_id)
            
            # Process based on file type
            document_type = ""
            output_path = PROCESSED_DIR / f"{file_id}.txt"
            
            if content_type == "application/pdf":
                document_type = "pdf"
                await self._process_pdf(file_path, output_path)
            elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                document_type = "docx"
                await self._process_docx(file_path, output_path)
            elif content_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
                document_type = "pptx"
                await self._process_pptx(file_path, output_path)
            else:
                # For other text-based formats, just copy
                document_type = "text"
                shutil.copy2(file_path, output_path)
            
            # Update metadata
            self.document_metadata[file_id].document_type = document_type
            self.document_metadata[file_id].status = "processed"
            self.save_metadata()
            
            # Reload documents in vector store
            await self.load_documents()
            
            # Broadcast document status update to all clients
            await self.broadcast_document_status(file_id)
            
            return True
        except Exception as e:
            print(f"Error processing document {file_id}: {e}")
            self.document_metadata[file_id].status = "error"
            self.save_metadata()
            # Broadcast error status
            await self.broadcast_document_status(file_id)
            return False
    
    async def _process_pdf(self, file_path: str, output_path: str):
        """Process a PDF file and convert to text"""
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            for page in pages:
                await f.write(page.page_content)
                await f.write("\n\n")
    
    async def _process_docx(self, file_path: str, output_path: str):
        """Process a DOCX file and convert to text"""
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                await f.write(doc.page_content)
    
    async def _process_pptx(self, file_path: str, output_path: str):
        """Process a PPTX file and convert to text"""
        loader = UnstructuredPowerPointLoader(file_path)
        documents = loader.load()
        
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                await f.write(doc.page_content)
                await f.write("\n\n")
    
    async def process_youtube(self, video_id: str, file_id: str):
        """Process a YouTube video by downloading and transcribing it"""
        try:
            # Update status
            self.document_metadata[file_id].status = "downloading"
            self.save_metadata()
            await self.broadcast_document_status(file_id)
            
            # Create temporary directory for processing
            temp_dir = TEMP_DIR / file_id
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download YouTube video
            yt = pytube.YouTube(f"https://www.youtube.com/watch?v={video_id}")
            video_stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
            
            if not video_stream:
                raise Exception("No suitable video stream found")
            
            video_path = video_stream.download(output_path=str(temp_dir))
            
            # Update metadata
            self.document_metadata[file_id].youtube_title = yt.title
            self.document_metadata[file_id].youtube_thumbnail = yt.thumbnail_url
            self.document_metadata[file_id].status = "transcribing"
            self.document_metadata[file_id].duration = yt.length
            self.save_metadata()
            await self.broadcast_document_status(file_id)
            
            # Transcribe video
            return await self.transcribe_video(video_path, file_id)
            
        except Exception as e:
            print(f"Error processing YouTube video: {e}")
            self.document_metadata[file_id].status = "error"
            self.save_metadata()
            await self.broadcast_document_status(file_id)
            return False
    
    async def process_video(self, file_path: str, file_id: str):
        """Process an uploaded video file by transcribing it"""
        try:
            # Update status
            self.document_metadata[file_id].status = "processing"
            self.save_metadata()
            await self.broadcast_document_status(file_id)
            
            # Get video duration
            probe = ffmpeg.probe(file_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            duration = int(float(video_info.get('duration', 0)))
            
            # Update metadata
            self.document_metadata[file_id].duration = duration
            self.document_metadata[file_id].status = "transcribing"
            self.save_metadata()
            await self.broadcast_document_status(file_id)
            
            # Transcribe video
            return await self.transcribe_video(file_path, file_id)
            
        except Exception as e:
            print(f"Error processing video: {e}")
            self.document_metadata[file_id].status = "error"
            self.save_metadata()
            await self.broadcast_document_status(file_id)
            return False
    
    async def transcribe_video(self, video_path: str, file_id: str):
        """Transcribe a video file using Whisper with timestamps"""
        try:
            # Transcribe with whisper
            result = self.whisper_model.transcribe(video_path, word_timestamps=True)
            
            # Extract segments with timestamps
            segments = []
            for segment in result.get("segments", []):
                segments.append(TranscriptSegment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"]
                ))
            
            # Save transcript to processed directory
            output_path = PROCESSED_DIR / f"{file_id}.txt"
            async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
                await f.write(f"# Transcript: {self.document_metadata[file_id].filename}\n\n")
                
                for segment in segments:
                    timestamp = f"[{self._format_timestamp(segment.start)} - {self._format_timestamp(segment.end)}]"
                    await f.write(f"{timestamp} {segment.text}\n\n")
            
            # Update metadata
            self.document_metadata[file_id].segments = segments
            self.document_metadata[file_id].status = "processed"
            self.save_metadata()
            
            # Reload documents
            await self.load_documents()
            
            # Broadcast document status update to all clients
            await self.broadcast_document_status(file_id)
            
            return True
        except Exception as e:
            print(f"Error transcribing video: {e}")
            self.document_metadata[file_id].status = "error"
            self.save_metadata()
            # Broadcast error status
            await self.broadcast_document_status(file_id)
            return False
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as mm:ss"""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its associated files from the system"""
        try:
            if document_id not in self.document_metadata:
                return False
                
            # Get document metadata
            doc_metadata = self.document_metadata[document_id]
            
            # Delete the original uploaded file if it exists
            original_file = None
            if doc_metadata.document_type in ["pdf", "docx", "pptx", "text"]:
                # Regular document
                original_file = list(UPLOAD_DIR.glob(f"{document_id}_*"))
            elif doc_metadata.youtube_id:
                # YouTube video - delete temp directory
                temp_dir = TEMP_DIR / document_id
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    
            # Delete the original file if found
            if original_file and len(original_file) > 0:
                original_file[0].unlink(missing_ok=True)
                
            # Delete the processed file
            processed_file = PROCESSED_DIR / f"{document_id}.txt"
            if processed_file.exists():
                processed_file.unlink()
                
            # Remove from metadata
            del self.document_metadata[document_id]
            self.save_metadata()
            
            # Reload documents to update vector store
            await self.load_documents()
            
            # Notify connected clients about the deletion
            for connection_id in manager.active_connections:
                try:
                    await manager.send_json(connection_id, {
                        "type": "document_deleted",
                        "document_id": document_id
                    })
                except Exception as e:
                    print(f"Error sending document deletion notification to {connection_id}: {e}")
                    
            return True
            
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False

# Initialize the RAG processor
rag_processor = RAGProcessor(db_dir=str(DB_DIR))

# API endpoints
@app.post("/api/documents/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload and process a document file (PDF, DOCX, PPTX, etc.)"""
    # Generate a unique ID
    file_id = str(uuid.uuid4())
    
    # Validate content type
    allowed_types = [
        "application/pdf",  # PDF
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # PPTX
        "text/plain",  # TXT
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    
    # Save file
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    
    # Create metadata
    metadata = DocumentMetadata(
        id=file_id,
        filename=file.filename,
        content_type=file.content_type,
        size=len(content),
        upload_time=datetime.now().isoformat(),
        document_type="unknown",  # Will be updated during processing
        status="uploaded"
    )
    
    rag_processor.document_metadata[file_id] = metadata
    rag_processor.save_metadata()
    
    # Process document in background
    background_tasks.add_task(
        rag_processor.process_document,
        file_id,
        str(file_path),
        file.filename,
        file.content_type
    )
    
    return metadata

@app.post("/api/videos/upload", response_model=VideoResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload and process a video file"""
    # Generate a unique ID
    file_id = str(uuid.uuid4())
    
    # Validate content type
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    
    # Save file
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    
    # Create metadata
    metadata = DocumentMetadata(
        id=file_id,
        filename=file.filename,
        content_type=file.content_type,
        size=len(content),
        upload_time=datetime.now().isoformat(),
        document_type="video",
        status="uploaded"
    )
    
    rag_processor.document_metadata[file_id] = metadata
    rag_processor.save_metadata()
    
    # Process video in background
    background_tasks.add_task(
        rag_processor.process_video,
        str(file_path),
        file_id
    )
    
    return VideoResponse(
        id=file_id,
        filename=file.filename,
        upload_time=metadata.upload_time,
        status=metadata.status
    )

@app.post("/api/youtube", response_model=YouTubeResponse)
async def process_youtube_video(
    background_tasks: BackgroundTasks,
    request: YouTubeRequest
):
    """Process a YouTube video URL"""
    # Extract video ID from URL
    url = str(request.url)
    video_id = None
    
    if "youtube.com/watch" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Generate a unique ID
    file_id = str(uuid.uuid4())
    
    # Create metadata
    metadata = DocumentMetadata(
        id=file_id,
        filename=f"youtube_{video_id}",
        content_type="video/youtube",
        size=0,  # Unknown until downloaded
        upload_time=datetime.now().isoformat(),
        document_type="youtube",
        status="pending",
        youtube_id=video_id
    )
    
    rag_processor.document_metadata[file_id] = metadata
    rag_processor.save_metadata()
    
    # Process YouTube video in background
    background_tasks.add_task(
        rag_processor.process_youtube,
        video_id,
        file_id
    )
    
    return YouTubeResponse(
        id=file_id,
        video_id=video_id,
        title="Processing...",  # Will be updated during processing
        upload_time=metadata.upload_time,
        status=metadata.status
    )

@app.get("/api/documents", response_model=List[DocumentMetadata])
async def list_documents():
    """List all processed documents"""
    return list(rag_processor.document_metadata.values())

@app.get("/api/documents/{document_id}", response_model=DocumentMetadata)
async def get_document(document_id: str):
    """Get a specific document's metadata"""
    if document_id not in rag_processor.document_metadata:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return rag_processor.document_metadata[document_id]

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its associated files"""
    if document_id not in rag_processor.document_metadata:
        raise HTTPException(status_code=404, detail="Document not found")
    
    success = await rag_processor.delete_document(document_id)
    
    if success:
        return {"status": "success", "message": f"Document {document_id} deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the RAG system using a REST API"""
    # Create or use existing session
    session_id = request.session_id or str(uuid.uuid4())
    
    # Initialize for session if needed
    if session_id not in rag_processor.conversation_chains:
        await rag_processor.initialize_for_connection(session_id)
    
    # Process query
    answer = await rag_processor.chat(session_id, request.query)
    
    return ChatResponse(answer=answer, session_id=session_id)

@app.post("/api/chat/clear")
async def clear_chat(session_id: str = Form(...)):
    """Clear chat history for a session"""
    if session_id in rag_processor.memories:
        rag_processor.clear_memory(session_id)
        return {"status": "success", "message": "Chat history cleared"}
    
    return {"status": "error", "message": "Session not found"}

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    connection_id = await manager.connect(websocket)
    
    try:
        # Send welcome message
        await manager.send_json(connection_id, {
            "type": "system",
            "content": f"Connected to RAG Chat. Loading documents..."
        })
        
        # Send current document status for all documents
        for doc_id, doc in rag_processor.document_metadata.items():
            await manager.send_json(connection_id, {
                "type": "document_update",
                "document": doc.dict()
            })
        
        # Create streaming handler for this connection
        streaming_handler = StreamingCallbackHandler(connection_id)
        
        # Initialize RAG app for this connection
        success = await rag_processor.initialize_for_connection(connection_id, streaming_handler)
        
        if not success:
            await manager.send_json(connection_id, {
                "type": "system",
                "content": "Failed to initialize chat. Please try again later."
            })
            return
        
        await manager.send_json(connection_id, {
            "type": "system",
            "content": "Ready! You can now chat with the RAG system."
        })
        
        # Process messages
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "message":
                query = message_data.get("content", "")
                
                # Process the query
                try:
                    # Send start message to indicate a new response is coming
                    await manager.send_json(connection_id, {"type": "start"})
                    
                    # Process the query (streaming will happen via the callback)
                    _ = await rag_processor.chat(connection_id, query)
                    
                    # Send end message to indicate the response is complete
                    await manager.send_json(connection_id, {"type": "end"})
                except Exception as e:
                    await manager.send_json(connection_id, {
                        "type": "system",
                        "content": f"Error: {str(e)}"
                    })
            
            elif message_data.get("type") == "command":
                command = message_data.get("command", "")
                
                if command == "clear":
                    if rag_processor.clear_memory(connection_id):
                        await manager.send_json(connection_id, {
                            "type": "system",
                            "content": "Chat history cleared"
                        })
                    else:
                        await manager.send_json(connection_id, {
                            "type": "system",
                            "content": "Failed to clear chat history"
                        })
                elif command == "delete_document":
                    document_id = message_data.get("document_id")
                    if not document_id:
                        await manager.send_json(connection_id, {
                            "type": "system",
                            "content": "Missing document_id for deletion"
                        })
                    elif document_id not in rag_processor.document_metadata:
                        await manager.send_json(connection_id, {
                            "type": "system",
                            "content": f"Document {document_id} not found"
                        })
                    else:
                        success = await rag_processor.delete_document(document_id)
                        if success:
                            await manager.send_json(connection_id, {
                                "type": "system",
                                "content": f"Document {document_id} deleted successfully"
                            })
                        else:
                            await manager.send_json(connection_id, {
                                "type": "system",
                                "content": f"Failed to delete document {document_id}"
                            })
    
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
        rag_processor.cleanup_connection(connection_id)
        print(f"Client #{connection_id} disconnected")
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await manager.send_json(connection_id, {
                "type": "system",
                "content": f"An error occurred: {str(e)}"
            })
        except:
            pass
        manager.disconnect(connection_id)
        rag_processor.cleanup_connection(connection_id)

@app.on_event("startup")
async def startup_event():
    await rag_processor.load_documents()

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api_server:app", host="0.0.0.0", port=8000, reload=True) 