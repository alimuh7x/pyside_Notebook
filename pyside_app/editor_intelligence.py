from __future__ import annotations

import builtins
import json
import keyword
import os
import shutil
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, Signal

from utils.notebook_eval import ALLOWED_NOTEBOOK_FUNCTIONS, PHYSICAL_CONSTANTS, UNIT_CONSTANTS


def build_completion_words(namespace: dict[str, Any] | None = None) -> list[str]:
    names: set[str] = set(keyword.kwlist)
    names.update(name for name in dir(builtins) if name.isidentifier())
    names.update(name for name in ALLOWED_NOTEBOOK_FUNCTIONS.keys() if str(name).isidentifier())
    names.update(name for name in PHYSICAL_CONSTANTS.keys() if str(name).isidentifier())
    names.update(name for name in UNIT_CONSTANTS.keys() if str(name).isidentifier())
    for name in (namespace or {}).keys():
        if str(name).isidentifier() and not str(name).startswith("__"):
            names.add(str(name))
    words = sorted(names)
    print(f"[debug][editor-intelligence] build_completion_words count={len(words)}", flush=True)
    return words


class NotebookLspClient(QObject):
    completions_ready = Signal(str, list)
    diagnostics_ready = Signal(str, list)
    availability_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None, command: str | None = None) -> None:
        super().__init__(parent)
        self._command = command or self.resolve_command()
        self._process: QProcess | None = None
        self._buffer = bytearray()
        self._request_id = 0
        self._pending_completions: dict[int, str] = {}
        self._document_versions: dict[str, int] = {}
        self._opened_documents: set[str] = set()
        self._initialized = False
        self.destroyed.connect(lambda *_args: self.shutdown())
        print(f"[debug][lsp-client] init command={self._command!r}", flush=True)

    @staticmethod
    def resolve_command() -> str | None:
        explicit = os.environ.get("NOTEBOOK_LSP_COMMAND")
        if explicit:
            print(f"[debug][lsp-client] resolve_command explicit={explicit!r}", flush=True)
            return explicit
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root / "myenv" / "bin" / "pylsp"
        if candidate.exists():
            print(f"[debug][lsp-client] resolve_command repo_candidate={str(candidate)!r}", flush=True)
            return str(candidate)
        found = shutil.which("pylsp")
        print(f"[debug][lsp-client] resolve_command which={found!r}", flush=True)
        return found

    def is_available(self) -> bool:
        available = self._command is not None
        print(f"[debug][lsp-client] is_available available={available}", flush=True)
        return available

    def start(self) -> None:
        if not self.is_available():
            self.availability_changed.emit(False)
            return
        if self._process is not None:
            return
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.finished.connect(self._handle_process_finished)
        self._process.start(self._command, [])
        started = self._process.waitForStarted(3000)
        print(f"[debug][lsp-client] start started={started}", flush=True)
        if not started:
            self._process = None
            self.availability_changed.emit(False)
            return
        self._send_request("initialize", {"processId": None, "rootUri": None, "capabilities": {}})
        self._send_notification("initialized", {})
        self._initialized = True
        self.availability_changed.emit(True)

    def shutdown(self) -> None:
        if self._process is None:
            print("[debug][lsp-client] shutdown skipped", flush=True)
            return
        print("[debug][lsp-client] shutdown start", flush=True)
        self._process.terminate()
        if not self._process.waitForFinished(1000):
            print("[debug][lsp-client] shutdown kill", flush=True)
            self._process.kill()
            self._process.waitForFinished(1000)
        self._process = None
        self._initialized = False
        try:
            self.availability_changed.emit(False)
        except RuntimeError as exc:
            print(f"[debug][lsp-client] shutdown emit_skipped error={exc!r}", flush=True)
        print("[debug][lsp-client] shutdown done", flush=True)

    def open_document(self, uri: str, text: str) -> None:
        if not self._initialized:
            return
        version = self._document_versions.get(uri, 0) + 1
        self._document_versions[uri] = version
        if uri not in self._opened_documents:
            print(f"[debug][lsp-client] open_document uri={uri!r} version={version}", flush=True)
            self._send_notification(
                "textDocument/didOpen",
                {"textDocument": {"uri": uri, "languageId": "python", "version": version, "text": text}},
            )
            self._opened_documents.add(uri)
            return
        self.change_document(uri, text)

    def change_document(self, uri: str, text: str) -> None:
        if not self._initialized or uri not in self._opened_documents:
            return
        version = self._document_versions.get(uri, 0) + 1
        self._document_versions[uri] = version
        print(f"[debug][lsp-client] change_document uri={uri!r} version={version}", flush=True)
        self._send_notification(
            "textDocument/didChange",
            {"textDocument": {"uri": uri, "version": version}, "contentChanges": [{"text": text}]},
        )

    def request_completion(self, uri: str, line: int, character: int) -> None:
        if not self._initialized or uri not in self._opened_documents:
            return
        request_id = self._send_request(
            "textDocument/completion",
            {"textDocument": {"uri": uri}, "position": {"line": line, "character": character}},
        )
        self._pending_completions[request_id] = uri
        print(
            f"[debug][lsp-client] request_completion uri={uri!r} line={line} character={character} id={request_id}",
            flush=True,
        )

    def _send_request(self, method: str, params: dict[str, Any]) -> int:
        self._request_id += 1
        payload = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}
        self._write_message(payload)
        return self._request_id

    def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        self._write_message(payload)

    def _write_message(self, payload: dict[str, Any]) -> None:
        if self._process is None:
            return
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        print(f"[debug][lsp-client] write_message method={payload.get('method')!r}", flush=True)
        self._process.write(header + body)
        self._process.waitForBytesWritten(1000)

    def _read_stdout(self) -> None:
        if self._process is None:
            return
        self._buffer.extend(bytes(self._process.readAllStandardOutput()))
        while True:
            separator = self._buffer.find(b"\r\n\r\n")
            if separator == -1:
                return
            header = self._buffer[:separator].decode("utf-8", errors="ignore")
            length_line = next((line for line in header.split("\r\n") if line.lower().startswith("content-length:")), None)
            if length_line is None:
                self._buffer = self._buffer[separator + 4 :]
                continue
            length = int(length_line.split(":", 1)[1].strip())
            message_start = separator + 4
            message_end = message_start + length
            if len(self._buffer) < message_end:
                return
            body = bytes(self._buffer[message_start:message_end])
            self._buffer = self._buffer[message_end:]
            self._handle_message(json.loads(body.decode("utf-8")))

    def _read_stderr(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardError()).decode("utf-8", errors="ignore").strip()
        if text:
            print(f"[debug][lsp-client] stderr text={text!r}", flush=True)

    def _handle_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        print(
            f"[debug][lsp-client] process_finished exit_code={exit_code} exit_status={int(exit_status)}",
            flush=True,
        )
        self._process = None
        self._initialized = False
        self.availability_changed.emit(False)

    def _handle_message(self, message: dict[str, Any]) -> None:
        print(f"[debug][lsp-client] handle_message keys={list(message.keys())}", flush=True)
        method = message.get("method")
        if method == "textDocument/publishDiagnostics":
            params = message.get("params") or {}
            self.diagnostics_ready.emit(params.get("uri", ""), list(params.get("diagnostics") or []))
            return
        if "id" in message and message.get("id") in self._pending_completions:
            uri = self._pending_completions.pop(message["id"])
            result = message.get("result") or {}
            items = result.get("items", result if isinstance(result, list) else [])
            labels = [item.get("label", "") for item in items if item.get("label")]
            print(f"[debug][lsp-client] completions_ready uri={uri!r} count={len(labels)}", flush=True)
            self.completions_ready.emit(uri, labels)
