"""推送通道自测：Bark(iOS 首选) / Server酱(微信) / PushDeer 连通性 + 送达。

只用标准库 urllib，不依赖第三方；从环境变量读密钥，绝不硬编码。
    BARK_KEY=<设备key>             Bark(iOS,免费不限量,首选)
    BARK_ENDPOINT=...              Bark 自建端点(可选,默认官方 api.day.app)
    SERVERCHAN_SCKEY=<你的SCKEY>   微信通知(Server酱 Turbo,免费版每天仅5条)
    PUSHDEER_KEY=<你的pushkey>     PushDeer(可选)
    PUSHDEER_ENDPOINT=...          PushDeer 自建端点(可选,默认官方)

做的事：
  1) 连通性探测：能否连上对应服务域名
  2) 每个已配置的通道真发一条测试消息，打印 HTTP 状态 + 返回内容
跑法（在容器或宿主机）：
    BARK_KEY=xxxx python3 realtime/notify_selftest.py
输出里绝不打印密钥本身。
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.parse
import urllib.request


def _probe(host: str, port: int = 443, timeout: float = 6.0) -> bool:
    """纯 TCP 连通性探测：能否建到 host:port 的连接。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            print(f"[probe][OK] 可连 {host}:{port}（公网可达）", flush=True)
            return True
    except Exception as e:  # noqa: BLE001
        print(f"[probe][FAIL] 连不上 {host}:{port} —— {type(e).__name__}: {e}", flush=True)
        return False


def _host_of(url: str) -> str:
    return url.split("//")[-1].split("/")[0]


def _post(url: str, data: dict, timeout: float = 10.0):
    """POST 表单，返回 (http_status, body_text)；异常返回 (None, 错误串)。"""
    try:
        payload = urllib.parse.urlencode(data).encode()
        req = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            return resp.status, body
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def main() -> int:
    # 复用引擎的 env 文件加载：让自测也能从 notify.env 读凭证（无需每次 -e 传）。
    # 脚本直跑时 sys.path[0] 是本文件目录，需把父目录(含 realtime 包)加入 path。
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from realtime.config import _load_env_file
        _load_env_file()
    except Exception:  # noqa: BLE001 - 读不到就退回纯环境变量
        pass

    bark_key = os.environ.get("BARK_KEY", "").strip()
    bark_endpoint = os.environ.get("BARK_ENDPOINT", "https://api.day.app").strip()
    sckey = os.environ.get("SERVERCHAN_SCKEY", "").strip()
    pushkey = os.environ.get("PUSHDEER_KEY", "").strip()
    endpoint = os.environ.get("PUSHDEER_ENDPOINT", "https://api2.pushdeer.com").strip()

    stamp = time.strftime("%H:%M:%S")
    title = "[实时层] 推送自测"
    body = (f"这是一条来自 A股实时层的测试消息（{stamp}）。\n"
            "看到本条即说明：容器→公网→手机 链路已通。")

    print("==== 推送通道自测（Bark / Server酱 / PushDeer）====", flush=True)
    print(f"配置：BARK_KEY={'已设置' if bark_key else '未设置'} "
          f"SERVERCHAN_SCKEY={'已设置' if sckey else '未设置'} "
          f"PUSHDEER_KEY={'已设置' if pushkey else '未设置'}", flush=True)

    if not (bark_key or sckey or pushkey):
        print("[skip] 三个密钥都没设置。请在命令前加环境变量再跑，例如：", flush=True)
        print("       BARK_KEY=你的设备key python3 realtime/notify_selftest.py", flush=True)
        return 2

    ok = True

    # 1) Bark（iOS 首选）
    if bark_key:
        print("\n---- [Bark] 连通性 + 发送 ----", flush=True)
        _probe(_host_of(bark_endpoint))
        url = f"{bark_endpoint.rstrip('/')}/{bark_key}"
        status, resp = _post(url, {"title": title, "body": body, "group": "A股实时"})
        if status is None:
            print(f"[bark][FAIL] 请求异常：{resp}", flush=True); ok = False
        else:
            print(f"[bark] HTTP {status}  返回：{resp[:300]}", flush=True)
            try:
                j = json.loads(resp)
                if status == 200 and j.get("code") in (200, 0):
                    print("[bark][OK] 已提交，请查看 iPhone 是否收到通知。", flush=True)
                else:
                    print(f"[bark][WARN] 返回 code={j.get('code')}，key 可能有误。",
                          flush=True); ok = False
            except Exception:  # noqa: BLE001
                print("[bark][WARN] 返回非 JSON，人工核对上面内容。", flush=True)

    # 2) Server酱（微信）
    if sckey:
        print("\n---- [Server酱] 连通性 + 发送（微信）----", flush=True)
        _probe("sctapi.ftqq.com")
        url = f"https://sctapi.ftqq.com/{sckey}.send"
        status, resp = _post(url, {"title": title, "desp": body})
        if status is None:
            print(f"[serverchan][FAIL] 请求异常：{resp}", flush=True); ok = False
        else:
            print(f"[serverchan] HTTP {status}  返回：{resp[:300]}", flush=True)
            try:
                j = json.loads(resp)
                if status == 200 and j.get("code") in (0, "0", None):
                    print("[serverchan][OK] 已提交，请查看微信是否收到。", flush=True)
                else:
                    print(f"[serverchan][WARN] 返回 code={j.get('code')}，"
                          "可能 key 无效/未关注方糖/超每日5条限额。", flush=True); ok = False
            except Exception:  # noqa: BLE001
                print("[serverchan][WARN] 返回非 JSON，人工核对上面内容。", flush=True)

    # 3) PushDeer（可选）
    if pushkey:
        print("\n---- [PushDeer] 发送 ----", flush=True)
        _probe(_host_of(endpoint))
        url = f"{endpoint.rstrip('/')}/message/push"
        status, resp = _post(url, {"pushkey": pushkey, "text": title,
                                   "desp": body, "type": "markdown"})
        if status is None:
            print(f"[pushdeer][FAIL] 请求异常：{resp}", flush=True); ok = False
        else:
            print(f"[pushdeer] HTTP {status}  返回：{resp[:300]}", flush=True)

    print(f"\n==== 自测结束：{'通道可用' if ok else '有失败项，见上'} ====", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

