from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import json
import os
import hashlib

app = FastAPI(
    title="User-Specific Knowledge Graph Server",
    version="1.3.0",
    description="A structured knowledge graph memory system that supports user-specific entity and relation storage, observation tracking, and manipulation.",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Persistence Setup -----
DATA_DIR = Path(os.getenv("MEMORY_DATA_PATH", "/app/data"))
DATA_DIR.mkdir(exist_ok=True)

def get_memory_file_path(user_id: str) -> Path:
    # Use a hash of the user_id for the filename to be safe
    hashed_user_id = hashlib.sha256(user_id.encode()).hexdigest()
    return DATA_DIR / f"memory_{hashed_user_id}.json"

# ----- Data Models -----
class Entity(BaseModel):
    name: str = Field(..., description="The name of the entity")
    entityType: str = Field(..., description="The type of the entity")
    observations: List[str] = Field(
        ..., description="An array of observation contents associated with the entity"
    )

class Relation(BaseModel):
    from_: str = Field(
        ...,
        alias="from",
        description="The name of the entity where the relation starts",
    )
    to: str = Field(..., description="The name of the entity where the relation ends")
    relationType: str = Field(..., description="The type of the relation")

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relations: List[Relation]

# ----- I/O Handlers -----
def read_graph_file(file_path: Path) -> KnowledgeGraph:
    if not file_path.exists():
        return KnowledgeGraph(entities=[], relations=[])
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
        entities = []
        relations = []
        for line in lines:
            item = json.loads(line)
            if item.get("type") == "entity":
                entities.append(Entity(**{k: v for k, v in item.items() if k != 'type'}))
            elif item.get("type") == "relation":
                relations.append(Relation(**{k: v for k, v in item.items() if k != 'type'}))
    return KnowledgeGraph(entities=entities, relations=relations)

def save_graph(graph: KnowledgeGraph, file_path: Path):
    lines = [json.dumps({"type": "entity", **e.dict()}) for e in graph.entities] + [
        json.dumps({"type": "relation", **r.dict(by_alias=True)})
        for r in graph.relations
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ----- Request Models -----
class UserSpecificRequest(BaseModel):
    user_id: str = Field(..., description="User ID for user-specific memory.")

class CreateEntitiesRequest(UserSpecificRequest):
    entities: List[Entity] = Field(..., description="List of entities to create")

class CreateRelationsRequest(UserSpecificRequest):
    relations: List[Relation] = Field(..., description="List of relations to create.")

class ObservationItem(BaseModel):
    entityName: str
    contents: List[str]

class AddObservationsRequest(UserSpecificRequest):
    observations: List[ObservationItem]

class DeletionItem(BaseModel):
    entityName: str
    observations: List[str]

class DeleteObservationsRequest(UserSpecificRequest):
    deletions: List[DeletionItem]

class DeleteEntitiesRequest(UserSpecificRequest):
    entityNames: List[str]

class DeleteRelationsRequest(UserSpecificRequest):
    relations: List[Relation]

class SearchNodesRequest(UserSpecificRequest):
    query: str

class OpenNodesRequest(UserSpecificRequest):
    names: List[str]

# ----- Endpoints -----
@app.post("/create_entities", summary="Create multiple entities in the graph")
def create_entities(req: CreateEntitiesRequest):
    file_path = get_memory_file_path(req.user_id)
    graph = read_graph_file(file_path)
    existing_names = {e.name for e in graph.entities}
    new_entities = [e for e in req.entities if e.name not in existing_names]
    graph.entities.extend(new_entities)
    save_graph(graph, file_path)
    return new_entities

@app.post("/create_relations", summary="Create multiple relations between entities")
def create_relations(req: CreateRelationsRequest):
    file_path = get_memory_file_path(req.user_id)
    graph = read_graph_file(file_path)
    existing = {(r.from_, r.to, r.relationType) for r in graph.relations}
    new = [r for r in req.relations if (r.from_, r.to, r.relationType) not in existing]
    graph.relations.extend(new)
    save_graph(graph, file_path)
    return new

@app.post("/add_observations", summary="Add new observations to existing entities")
def add_observations(req: AddObservationsRequest):
    file_path = get_memory_file_path(req.user_id)
    graph = read_graph_file(file_path)
    results = []
    for obs in req.observations:
        entity = next((e for e in graph.entities if e.name == obs.entityName), None)
        if not entity:
            raise HTTPException(status_code=404, detail=f"Entity {obs.entityName} not found")
        added = [c for c in obs.contents if c not in entity.observations]
        entity.observations.extend(added)
        results.append({"entityName": obs.entityName, "addedObservations": added})
    save_graph(graph, file_path)
    return results

@app.post("/delete_entities", summary="Delete entities and associated relations")
def delete_entities(req: DeleteEntitiesRequest):
    file_path = get_memory_file_path(req.user_id)
    graph = read_graph_file(file_path)
    graph.entities = [e for e in graph.entities if e.name not in req.entityNames]
    graph.relations = [r for r in graph.relations if r.from_ not in req.entityNames and r.to not in req.entityNames]
    save_graph(graph, file_path)
    return {"message": "Entities deleted successfully"}

@app.post("/delete_observations", summary="Delete specific observations from entities")
def delete_observations(req: DeleteObservationsRequest):
    file_path = get_memory_file_path(req.user_id)
    graph = read_graph_file(file_path)
    for deletion in req.deletions:
        entity = next((e for e in graph.entities if e.name == deletion.entityName), None)
        if entity:
            entity.observations = [obs for obs in entity.observations if obs not in deletion.observations]
    save_graph(graph, file_path)
    return {"message": "Observations deleted successfully"}

@app.post("/delete_relations", summary="Delete relations from the graph")
def delete_relations(req: DeleteRelationsRequest):
    file_path = get_memory_file_path(req.user_id)
    graph = read_graph_file(file_path)
    del_set = {(r.from_, r.to, r.relationType) for r in req.relations}
    graph.relations = [r for r in graph.relations if (r.from_, r.to, r.relationType) not in del_set]
    save_graph(graph, file_path)
    return {"message": "Relations deleted successfully"}

@app.get("/read_graph", response_model=KnowledgeGraph, summary="Read knowledge graph for a user")
def read_graph(user_id: str):
    file_path = get_memory_file_path(user_id)
    return read_graph_file(file_path)

@app.post("/search_nodes", response_model=KnowledgeGraph, summary="Search for nodes by keyword for a user")
def search_nodes(req: SearchNodesRequest):
    file_path = get_memory_file_path(req.user_id)
    graph = read_graph_file(file_path)
    entities = [e for e in graph.entities if req.query.lower() in e.name.lower() or req.query.lower() in e.entityType.lower() or any(req.query.lower() in o.lower() for o in e.observations)]
    names = {e.name for e in entities}
    relations = [r for r in graph.relations if r.from_ in names and r.to in names]
    return KnowledgeGraph(entities=entities, relations=relations)

@app.post("/open_nodes", response_model=KnowledgeGraph, summary="Open specific nodes by name for a user")
def open_nodes(req: OpenNodesRequest):
    file_path = get_memory_file_path(req.user_id)
    graph = read_graph_file(file_path)
    entities = [e for e in graph.entities if e.name in req.names]
    names = {e.name for e in entities}
    relations = [r for r in graph.relations if r.from_ in names and r.to in names]
    return KnowledgeGraph(entities=entities, relations=relations)
