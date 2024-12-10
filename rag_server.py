from flask import Flask, request, Response, render_template
from app import stream_response

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

chat_histories = {}
@app.route("/stream", methods=["POST"])
def stream():
  data = request.json
  user_query = data["query"]
  user_id = data["user_id"]

  if not user_id:
    return Response(status=400, response="User ID is required")

  if not user_id in chat_histories:
    chat_histories[user_id] = []

  def generate():
    full_response = ""  
    for chunk in stream_response(user_query, chat_histories[user_id]):
      full_response += chunk
      yield f"data: {chunk}\n\n"
      
    chat_histories[user_id].append({"role": "user", "content": user_query})
    chat_histories[user_id].append({"role": "assistant", "content": full_response})

  return Response(generate(), content_type="text/event-stream")

if __name__ == "__main__":
  app.run(debug=True, port=5001, host="0.0.0.0")