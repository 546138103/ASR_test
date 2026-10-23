import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from semantic_client import request_willingness, SemanticServiceError

class ClientTests(unittest.TestCase):
    def test_unicode_json_round_trip_and_service_token(self):
        seen = {}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen['path'] = self.path
                seen['auth'] = self.headers.get('Authorization')
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.update(body)
                data = json.dumps({'result': 'yes', 'reason': '表达参与意愿',
                                   'evidence': body['child_answer'], 'system_prompt': body['system_prompt'],
                                   'model': 'fixture'}, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *a): pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = request_willingness(' 好呀\n我想玩 ', server_url=f'http://127.0.0.1:{server.server_port}',
                                         system_prompt='你想和他们一起玩吗？', token='local-test')
            self.assertEqual(result['result'], 'yes')
            self.assertEqual(result['evidence'], ' 好呀\n我想玩 ')
            self.assertEqual(seen['path'], '/v1/tasks/willingness')
            self.assertEqual(seen['auth'], 'Bearer local-test')
            self.assertEqual(set(seen)-{'path','auth'}, {'child_answer','system_prompt'})
        finally:
            server.shutdown(); server.server_close(); worker.join()

    def test_bad_server_address_is_a_handled_configuration_error(self):
        for url in ('http://[::1', 'http://127.0.0.1:bad', 'http://127.0.0.1:99999'):
            with self.subTest(url=url), self.assertRaises(SemanticServiceError):
                request_willingness('想', server_url=url)

    def test_remote_plain_http_is_rejected_before_sending(self):
        with self.assertRaisesRegex(SemanticServiceError, 'HTTPS'):
            request_willingness('想', server_url='http://example.com')

    def test_unreachable_server_is_not_semantic_result(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), BaseHTTPRequestHandler)
        port = server.server_port
        server.server_close()
        with self.assertRaises(SemanticServiceError):
            request_willingness('想', server_url=f'http://127.0.0.1:{port}', timeout=.2)

if __name__ == '__main__': unittest.main()
