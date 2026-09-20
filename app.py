import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Import from the existing codebase without changing it
from main import Translator, find_checkpoint, get_config, PROJECT_DIR

# Initialize the model at startup
print("Initializing model... (this may take a moment)")
try:
    config = get_config()
    checkpoint_path = find_checkpoint(PROJECT_DIR / config["model_folder"], None)
    translator = Translator(checkpoint_path, force_cpu=False)
    print(f"Model loaded successfully from {checkpoint_path.name}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you have a trained model checkpoint in the 'weights' folder.")
    translator = None


class TranslationHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def do_GET(self):
        # Serve index.html for the root path
        if self.path == '/' or self.path == '/index.html':
            self.path = '/index.html'
            return SimpleHTTPRequestHandler.do_GET(self)
        
        # Deny access to other files for security, or just allow them
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path == '/api/translate':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_error(400, "Empty request body")
                return

            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                text = data.get('text', '').strip()
                
                if not text:
                    self._send_error(400, "No text provided")
                    return
                
                if not translator:
                    self._send_error(500, "Model is not loaded")
                    return

                # Translate the text
                translation, truncated_tokens = translator.translate(text)
                
                # Extract tokens for visualization
                source_ids = translator.source_tokenizer.encode(text).ids[:translator.max_source_tokens]
                encoder_tokens = [translator.source_sos] + source_ids + [translator.source_eos]
                encoder_tokens_str = [translator.source_tokenizer.id_to_token(i) for i in encoder_tokens]
                
                target_ids = translator.target_tokenizer.encode(translation).ids
                decoder_tokens = [translator.target_sos] + target_ids + [translator.target_eos]
                decoder_tokens_str = [translator.target_tokenizer.id_to_token(i) for i in decoder_tokens]

                # Extract attention matrices from the model
                # The model has N layers (usually 6).
                # Each attention_scores tensor is (1, h, seq_len, seq_len) or (1, h, tgt_len, src_len).
                def get_attn(blocks, name):
                    scores = []
                    for block in blocks:
                        attn = getattr(block, name).attention_scores[0] # (h, len1, len2)
                        scores.append(attn.cpu().tolist())
                    return scores
                
                # We only need the attention up to the token lengths
                # Since the decoder didn't see the final EOS token during its last step, its attention matrix might be 1 smaller.
                
                # slice the attention matrices to only the relevant tokens
                encoder_attn = []
                for block in translator.model.encoder.layers:
                    attn = block.self_attention_block.attention_scores[0] # (h, max_seq, max_seq)
                    # For encoder, the sequence length was padded to max_seq_len, we slice it to actual src_len
                    src_len = min(len(encoder_tokens_str), attn.shape[1])
                    attn_sliced = attn[:, :src_len, :src_len]
                    encoder_attn.append(attn_sliced.cpu().tolist())
                    
                decoder_attn = []
                cross_attn = []
                for block in translator.model.decoder.layers:
                    # self attention is (h, tgt_len, tgt_len)
                    d_attn = block.self_attention_block.attention_scores[0]
                    t_len = min(len(decoder_tokens_str), d_attn.shape[1])
                    decoder_attn.append(d_attn[:, :t_len, :t_len].cpu().tolist())
                    
                    c_attn = block.cross_attention_block.attention_scores[0]
                    cross_attn.append(c_attn[:, :t_len, :src_len].cpu().tolist())

                response_data = {
                    'translation': translation,
                    'truncated': truncated_tokens > 0,
                    'truncated': truncated_tokens > 0,
                    'visualization': {
                        'encoder_tokens': encoder_tokens_str,
                        'decoder_tokens': decoder_tokens_str,
                        'encoder_attn': encoder_attn,
                        'decoder_attn': decoder_attn,
                        'cross_attn': cross_attn,
                        'num_layers': len(encoder_attn),
                        'num_heads': len(encoder_attn[0]) if encoder_attn else 0
                    }
                }
                
                self._send_response(200, response_data)
                
            except json.JSONDecodeError:
                self._send_error(400, "Invalid JSON")
            except Exception as e:
                self._send_error(500, str(e))
        else:
            self._send_error(404, "Endpoint not found")

    def _send_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_error(self, status_code, message):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': message}).encode('utf-8'))


def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, TranslationHandler)
    print(f"\n=======================================================")
    print(f"UI Server is running! Open your browser and go to:")
    print(f"http://localhost:{port}")
    print(f"=======================================================\n")
    print("Press Ctrl+C to stop the server.")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == '__main__':
    # Ensure we run from the project directory so index.html is found
    os.chdir(PROJECT_DIR)
    run_server(port=8080)

