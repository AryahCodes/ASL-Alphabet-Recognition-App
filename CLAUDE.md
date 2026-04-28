## Global Working Rules

- Use subagents proactively to keep the main chat lightweight and organized.
- Prefer subagents for code review, debugging, codebase search, test generation, UI review, ML pipeline review, and planning larger edits.
- Do not dump long file contents, repeated search results, or unnecessary implementation details into the main chat.
- Summarize subagent findings in the main chat with only the files checked, recommendation, risks, and tests needed.
- For simple one-file edits or quick questions, answer directly without spawning unnecessary subagents.
- Before larger changes, use the most relevant subagent first, then summarize the plan before editing.

## Project Rules

- This project uses React, JavaScript, Python, Flask, Flask-SocketIO, MediaPipe, TensorFlow/Keras or TFLite, Socket.IO, Vercel, and Render.
- Do not assume Next.js, FastAPI, TypeScript, Firebase, SQLAlchemy, Flask-Migrate, Marshmallow, Flask-RESTful, uv, or a database unless the codebase clearly uses them.
- Follow the existing `backend/` and `frontend/` structure.
- Keep changes small and consistent with the current file organization.
- Do not introduce new dependencies unless asked.
- Do not rewrite the frontend architecture, backend architecture, ML pipeline, or model architecture unless asked.
- Be careful with webcam handling, Socket.IO events, frame buffering, prediction smoothing, model loading, and Render memory limits.
- Before editing, explain which files will change and why.
- After editing, summarize what changed and give the local test command.
- Do not commit or push unless explicitly asked.