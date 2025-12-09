import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import struct

model = SentenceTransformer('all-MiniLM-L6-v2')

# - Get current time: `{"name": "get_time", "arguments": {}}`
# - Search Google: `{"name": "google_search", "arguments": {"query": "your search query"}}`
# - Get weather forecast: `{"name": "weather", "arguments": {"location": "city, state/country"}}`
# - Search Wikipedia: `{"name": "wikipedia", "arguments": {"query": "your search query"}}`
# - Parse a URL: `{"name": "parse_url", "arguments": {"url": "URL to parse"}}`
# - Set a timer: `{"name": "set_timer", "arguments": {"duration": seconds}}`
# - Set an alarm: `{"name": "set_alarm", "arguments": {"time": "HH:mm", "label": "alarm name"}}`
# - Control device hardware:
#     - Open camera: `{"name": "device_hardware", "arguments": {"action": "open_camera"}}`
#     - Turn flashlight on/off: `{"name": "device_hardware", "arguments": {"action": "toggle_flashlight", "enable": true/false}}`
#     - Get battery level: `{"name": "device_hardware", "arguments": {"action": "get_battery_level"}}`

knowledge_base = [
    '{"name": "get_time", "description": "Get current time", "arguments": {}}',
    '{"name": "google_search", "description": "Search the web", "arguments": {"query": "string"}}',
    '{"name": "weather", "description": "Get weather forecast", "arguments": {"location": "city, state"}}',
    '{"name": "wikipedia", "description": "Search Wikipedia", "arguments": {"query": "string"}}',
    '{"name": "parse_url", "description": "Read content of a URL", "arguments": {"url": "url_string"}}',
    '{"name": "set_timer", "description": "Set a countdown timer", "arguments": {"duration": "seconds_int"}}',
    '{"name": "set_alarm", "description": "Set an alarm clock", "arguments": {"time": "HH:mm", "label": "string"}}',
    '{"name": "device_hardware", "description": "Open camera app", "arguments": {"action": "open_camera"}}',
    '{"name": "device_hardware", "description": "Toggle flashlight", "arguments": {"action": "toggle_flashlight", "enable": "boolean"}}',
    '{"name": "device_hardware", "description": "Get battery level", "arguments": {"action": "get_battery_level"}}'
]

key_words = [
    "current time clock what time is it tell me the time hour minute",
    "google search web internet find look up search for online browse query results",
    "weather forecast temperature rain hot cold sunny snow conditions outside climate",
    "wikipedia wiki encyclopedia define who is what is info information about learn about",
    "url parse website link read page summarize site content view web address",
    "set timer countdown stopwatch count down timer for minutes seconds",
    "set alarm wake up reminder alarm clock wake me up at schedule alert",
    "camera photo picture selfie open camera take a photo capture image lens",
    "flashlight light torch lamp lantern turn on light flashlight off dark see",
    "battery power level charge percentage energy how much battery juice status"
]

def serialize_vector(vector):
    return vector.tobytes()

filter_tags = [
    "time",         # get_time
    "google",       # google_search
    "weather",      # weather
    "wiki",         # wikipedia
    "url",          # parse_url
    "timer",        # set_timer
    "alarm",        # set_alarm
    "camera",       # device_hardware (camera)
    "flashlight",   # device_hardware (flashlight)
    "battery"       # device_hardware (battery)
]

def create_vector_db():
    conn = sqlite3.connect('friday_tools_android.db')
    c = conn.cursor()
    
    # Add 'filter_tag' column
    c.execute('''
              CREATE TABLE IF NOT EXISTS tools (
                id TEXT PRIMARY KEY,
                content TEXT,
                keywords TEXT,
                filter_tag TEXT,  
                embedding BLOB
              )
    ''')
    
    # Loop with ZIP including filter_tags
    for idx, (template, keyword, tag) in enumerate(zip(knowledge_base, key_words, filter_tags)):
        
        embedding = model.encode(keyword)
        embedding_blob = serialize_vector(np.array(embedding, dtype=np.float32))
        
        # Insert TAG
        c.execute("INSERT OR REPLACE INTO tools VALUES (?, ?, ?, ?, ?)",
                  (str(idx), template, keyword, tag, embedding_blob))
    
    conn.commit()
    conn.close()
    print("Database created with Hybrid Search tags.")

create_vector_db()