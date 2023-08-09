# This code defines a Flask application that can be used to interact with two chatbots:
#   * A PDF chatbot that can answer questions about PDF files.
#   * A Musk chatbot that can answer questions as if it were Elon Musk tweeting.

# The code first imports the necessary libraries, including Flask, SQLAlchemy, and json.

# Then, the code defines the Flask application and sets the CORS headers for the response.

# Next, the code defines two routes:
#   * `/api/bot4pdf` - This route is used to interact with the PDF chatbot.
#   * `/api/bot` - This route is used to interact with the Musk chatbot.

# The `/api/bot4pdf` route takes a message as input and returns a response from the PDF chatbot.
# The `/api/bot` route takes a message as input and returns a response from the Musk chatbot.

# Finally, the code defines the main function, which starts the Flask application.


import json
from sqlalchemy import create_engine, text
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

# import logging
# import sys


from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


# Specify the path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")

# Load the environment variables
load_dotenv(dotenv_path)

app = Flask(__name__)
debugMode = True

# Get the size of the server
# server_size = sys.getsizeof(app)

# Allow multiple origins
allowed_origins = ["https://npacts-dev.vercel.app", "http://localhost:3000"]

# # Enable Console Logging
# app.debug = os.environ.get("debug")
# logging.basicConfig(
#     level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s"
# )
# app.logger.debug("The server is {} bytes in size".format(server_size))

# # Add a handler to log messages to the console
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)
# app.logger.addHandler(console_handler)

if app.debug:
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
else:
    debugMode = False
    # Restrict the allowed origin in production
    CORS(
        app,
        resources={r"/api/*": {"origins": allowed_origins}},
        supports_credentials=True,
    )
app.config["CORS_HEADERS"] = "Content-Type"
# Set CORS headers for all routes


def row2dict(row):
    """
    This function converts a row from the MySQL database to a dictionary.

    Args:
        row: The row from the MySQL database.

    Returns:
        A dictionary with the data from the row.
    """

    d = {}
    for column in row.__table__.columns:
        d[column.name] = str(getattr(row, column.name))

    return d


@app.route("/api/upload", methods=["POST"])
def pdf_upload():
    # Set CORS headers for the response
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:3000",
        "Access-Control-Allow-Methods": "POST",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Credentials": "true",
    }
    pdf_file = request.files.get("file", "")
    if pdf_file is None or pdf_file.filename == "":
        return "No file selected"
    filename = secure_filename("uploaded.pdf")
    pdf_file.save(filename)

    return "Saved successfully"


@app.route("/api", methods=["GET"])
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/api/bot4pdf", methods=["POST"])
def pdf_bot():
    try:
        # Set CORS headers for the response
        headers = {
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Credentials": "true",
        }

        # Get the request data
        request_json = request.get_json()
        message = request_json["message"]

        # extract the text
        pdf_reader = PdfReader("uploaded.pdf")
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        if message:
            docs = knowledge_base.similarity_search(message)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=message)
            res = [{"response": response}]
            result = {"results": res}
            # Convert the dictionary to a JSON format
            json_result = json.dumps(result, ensure_ascii=False)

            # Return the response with CORS headers
            return (json_result, 200, headers)
        else:
            return jsonify({"error": str(ex)})
    except Exception as ex:
        # Handle any errors that may occur during processing
        # Return an error response if needed
        return jsonify({"error": str(ex)}), 500


@app.route("/api/bot", methods=["POST"])
def musk_bot():
    # initializing the db connection
    try:
        # Set CORS headers for the response
        headers = {
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Credentials": "true",
        }

        # Get the request data
        request_json = request.get_json()
        message = request_json["message"]

        # mindsdb is a MySQL db so these are the credentials
        user = os.environ.get("user")
        password = os.environ.get("password")
        host = os.environ.get("host")
        port = os.environ.get("port")
        database = os.environ.get("database")

        # Log the values to the console
        app.logger.debug("User: %s", user)
        app.logger.debug("Host: %s", host)
        app.logger.debug("Port: %s", port)
        app.logger.debug("Database: %s", database)

        # initializing the db connection
        def get_connection(user, password, host, port, database):
            return create_engine(
                url="mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(
                    user, password, host, port, database
                )
            )

        try:
            engine = get_connection(user, password, host, port, database)
            app.logger.debug("Connection to mindsdb created successfully.")
            print(f"Connection to {host} for user {user} created successfully.")
        except Exception as ex:
            app.logger.error("Failed to create engine: %s", ex)
            print("Failed to create engine due to the following error: \n", ex)
            return jsonify({"error": str(ex)}), 500

        # Run the query
        with engine.connect() as eng:
            sql_query = text(
                f"SELECT response FROM mindsdb.dev_chat WHERE text = '{message}' AND author_username = 'nicholas'"
            )
            query = eng.execute(sql_query)
            app.logger.debug("Query: %s", query)

            results = []
            for (response,) in query.fetchall():
                results.append({"response": response})
            app.logger.debug("Results: %s", results)

            # Check the structure of each dictionary in the results list
            for idx, result in enumerate(results):
                if not isinstance(result, dict):
                    app.logger.error("Invalid dictionary at index %d: %s", idx, result)
                    # You can also consider removing the problematic element from the results list

            # Create a dictionary to store the results
            result_dict = {"results": results}

            # Convert the dictionary to a JSON format
            json_result = json.dumps(result_dict, ensure_ascii=False, default=str)

            # Return the response with CORS headers
            return (json_result, 200, headers)

    except Exception as ex:
        # Handle any errors that may occur during processing
        # Return an error response if needed
        return jsonify({"error": str(ex)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=debugMode, threaded=True)
