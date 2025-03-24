import re
import openai
from collections import defaultdict

# OpenAI API Key (Replace with your actual API key)
openai.api_key = "your_openai_api_key"

def process_transcript(transcript):
    """
    Processes the transcript into timestamped segments.
    Assumes the transcript has timestamps in the format: [hh:mm:ss] text.
    Returns a list of dictionaries with 'timestamp' and 'text'.
    """
    segments = []
    for line in transcript.split("\n"):
        match = re.match(r"\[(\d+:\d+:\d+)\] (.+)", line)
        if match:
            timestamp, text = match.groups()
            segments.append({"timestamp": timestamp, "text": text})
    return segments

def extract_topics(segments):
    """
    Uses OpenAI API to extract key topics from the transcript in batches.
    """
    if not segments:
        return []

    texts = [segment['text'] for segment in segments]
    prompt = "Extract key topics from the following lecture texts:\n\n" + "\n".join(texts)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        extracted_topics = response["choices"][0]["message"]["content"].strip().split("\n")
        
        for segment, topic in zip(segments, extracted_topics):
            segment["topics"] = topic
    except Exception as e:
        print(f"Error fetching topics: {e}")
        for segment in segments:
            segment["topics"] = "Unknown"
    
    return segments

def build_topic_index(segments):
    """
    Creates a dictionary mapping topics to their corresponding timestamps.
    """
    topic_index = defaultdict(list)
    for segment in segments:
        topics = segment["topics"].split(", ")  # Assuming topics are comma-separated
        for topic in topics:
            topic_index[topic.lower()].append(segment["timestamp"])  
    return topic_index

def find_topic_timestamp(user_query, topic_index):
    """
    Finds the best-matching topic in the index and returns timestamps.
    """
    query_lower = user_query.lower()
    for topic in topic_index:
        if query_lower in topic:
            return topic_index[topic]
    return "Topic not found."

# Example transcript with timestamps
transcript = """
[00:05:10] Today we will discuss Newton's Laws of Motion.
[00:10:30] The First Law states that an object at rest stays at rest unless acted upon.
[00:15:45] The Second Law relates force, mass, and acceleration.
[00:20:20] The Third Law states that every action has an equal and opposite reaction.
"""

if __name__ == "__main__":
    # Process and extract topics
    segments = process_transcript(transcript)
    segments = extract_topics(segments)
    topic_index = build_topic_index(segments)

    # Simulate chatbot query
    user_query = "Newton's Third Law"
    timestamps = find_topic_timestamp(user_query, topic_index)
    print(f"Topic found at: {timestamps}")
