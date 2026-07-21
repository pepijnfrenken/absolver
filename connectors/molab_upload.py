"""Upload a local file to the Molab remote filesystem by chunking base64."""
import sys, os, base64, time
sys.path.insert(0, os.path.dirname(__file__))
from molab_exec import molab_exec

CHUNK_SIZE = 8000  # keep each API call under shell arg limit

def upload_file(url: str, token: str, local_path: str, remote_path: str, binary: bool = True):
    """Upload local_path to remote_path in chunks."""
    with open(local_path, 'rb' if binary else 'r') as f:
        data = f.read()
    
    if binary:
        raw = base64.b64encode(data).decode()
    else:
        raw = data

    # Create remote file by writing first chunk, then appending
    first = True
    for i in range(0, len(raw), CHUNK_SIZE):
        chunk = raw[i:i+CHUNK_SIZE]
        escaped = chunk.replace("'", "'\\''")
        if first:
            code = f"open('{remote_path}','w').write('{escaped}')"
            first = False
        else:
            code = f"with open('{remote_path}','a') as f: f.write('{escaped}')"
        
        result = molab_exec(url, token, code)
        if 'FAILED' in result:
            print(f'Chunk {i//CHUNK_SIZE} FAILED: {result[:200]}')
            return False
        print(f'Chunk {i//CHUNK_SIZE}: {len(chunk)}B → {result.strip()[:30]}')
    
    # If it's binary base64, decode it
    if binary:
        code = f"import base64; open('{remote_path}','wb').write(base64.b64decode(open('{remote_path}','r').read()))"
        result = molab_exec(url, token, code)
        print(f'Decode: {result.strip()[:60]}')
    
    return True

if __name__ == '__main__':
    upload_file(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], binary='--text' not in sys.argv)
