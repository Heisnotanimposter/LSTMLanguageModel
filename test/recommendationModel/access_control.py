# access_control.py

from flask import Flask, request, jsonify
from functools import wraps
import jwt
import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Use a secure key in production

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')  # Expecting 'Bearer <token>'
        if not token:
            return jsonify({'message': 'Token is missing'}), 403
        try:
            token = token.split()[1]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['user']
        except Exception as e:
            return jsonify({'message': 'Token is invalid', 'error': str(e)}), 403
        return f(current_user, *args, **kwargs)
    return decorated

@app.route('/login')
def login():
    # Authenticate user
    auth = request.authorization
    if auth and auth.username == 'user1' and auth.password == 'pass1':
        token = jwt.encode({
            'user': auth.username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
        }, app.config['SECRET_KEY'])
        return jsonify({'token': token})
    return jsonify({'message': 'Could not verify'}), 401

@app.route('/secure-data')
@token_required
def secure_data(current_user):
    return jsonify({'data': f'This is secure data for {current_user}'})

if __name__ == "__main__":
    app.run(ssl_context='adhoc')