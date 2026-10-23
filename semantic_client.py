"""Text-only HTTP client. No vendor SDK and no ARK_API_KEY needed."""
import ipaddress
from http.client import HTTPException
import json
import os
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

DEFAULT_SERVER = 'http://127.0.0.1:8765'

class SemanticServiceError(RuntimeError):
    pass

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_willingness(text, *, server_url=None, system_prompt=None, token=None, timeout=120):
    if not isinstance(text, str) or len(text) > 4096:
        raise SemanticServiceError('转写文字必须为字符串，且不超过 4096 字')
    base = (server_url or os.getenv('SEMANTIC_SERVER_URL') or DEFAULT_SERVER).rstrip('/')
    try:
        parts = urlsplit(base)
        parts.port  # Validate numeric range before constructing a request.
    except ValueError:
        raise SemanticServiceError('服务器地址或端口无效') from None
    try:
        local = ipaddress.ip_address(parts.hostname or '').is_loopback
    except ValueError:
        local = parts.hostname == 'localhost'
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise SemanticServiceError('服务器地址必须是有效的 HTTP(S) 地址，不包含账号、查询参数或片段')
    if not local and parts.scheme != 'https':
        raise SemanticServiceError('远程服务器必须使用 HTTPS，避免明文传输文字和访问令牌')
    body = {'child_answer': text}
    if system_prompt is not None:
        body['system_prompt'] = system_prompt
    headers = {'Content-Type': 'application/json; charset=utf-8', 'Accept': 'application/json'}
    service_token = os.getenv('SEMANTIC_SERVICE_TOKEN', '') if token is None else token
    if service_token:
        headers['Authorization'] = 'Bearer ' + service_token
    request = Request(base + '/v1/tasks/willingness', data=json.dumps(body, ensure_ascii=False).encode('utf-8'), headers=headers, method='POST')
    handlers = [NoRedirect()]
    if local:
        # Local IPC must not be routed to a system HTTP proxy.
        handlers.append(ProxyHandler({}))
    try:
        with build_opener(*handlers).open(request, timeout=timeout) as response:
            data = response.read(65537)
            if len(data) > 65536:
                raise SemanticServiceError('服务器响应过大')
            result = json.loads(data.decode('utf-8'))
    except HTTPError as error:
        explanations = {401: '服务访问令牌无效', 403: '服务器拒绝此客户端',
                        422: '文字或本轮提示格式不正确', 429: '服务器繁忙',
                        502: '服务器调用语义模型失败', 503: '服务器尚未配置 API Key'}
        raise SemanticServiceError(f'语义服务 HTTP {error.code}：' + explanations.get(error.code, '请求失败，检查服务器地址与配置')) from None
    except (URLError, TimeoutError, socket.timeout, OSError, HTTPException):
        raise SemanticServiceError('无法连接语义服务器或请求超时；请先启动服务器并检查地址') from None
    except (UnicodeError, ValueError):
        raise SemanticServiceError('服务器未返回有效 JSON') from None
    if not isinstance(result, dict) or result.get('result') not in ('yes', 'no', 'unclear') or not isinstance(result.get('reason'), str):
        raise SemanticServiceError('服务器返回的语义结果格式无效')
    if result.get('evidence') != text:
        raise SemanticServiceError('服务器响应与本次转写文字不匹配')
    return result
