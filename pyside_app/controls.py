from __future__ import annotations

from PySide6.QtCore import QModelIndex, Qt, Signal
from PySide6.QtWidgets import QComboBox, QWidget


class AutoCloseComboBox(QComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][auto-close-combo] init", flush=True)
        self.view().setMouseTracking(True)
        self.view().viewport().setMouseTracking(True)
        print("[debug][auto-close-combo] mouse_tracking enabled=True", flush=True)
        self.activated.connect(self._on_activated)
        self.currentIndexChanged.connect(self._on_index_changed)

    def _on_activated(self, index: int) -> None:
        print(f"[debug][auto-close-combo] activated index={index}", flush=True)

    def _on_index_changed(self, index: int) -> None:
        print(f"[debug][auto-close-combo] current_index_changed index={index}", flush=True)


class CheckableComboBox(QComboBox):
    checkedItemsChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][checkable-combo] init", flush=True)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Select Y variable(s)")
        self.view().setMouseTracking(True)
        self.view().viewport().setMouseTracking(True)
        print("[debug][checkable-combo] mouse_tracking enabled=True", flush=True)
        self.view().clicked.connect(self._toggle_clicked_item)
        self._update_display_text()

    def add_check_item(self, text: str, data: object, checked: bool = False) -> None:
        print(f"[debug][checkable-combo] add_item text={text!r} checked={checked}", flush=True)
        self.addItem(text, data)
        item = self.model().item(self.count() - 1, 0)
        if item is None:
            return
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        self._update_display_text()
        if checked and not self.signalsBlocked():
            print("[debug][checkable-combo] add_item:emit_checked_changed", flush=True)
            self.checkedItemsChanged.emit()

    def checked_values(self) -> list[str]:
        values: list[str] = []
        for index in range(self.count()):
            item = self.model().item(index, 0)
            if item is None:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                value = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(value, str):
                    values.append(value)
        print(f"[debug][checkable-combo] checked_values values={values!r}", flush=True)
        return values

    def set_checked_values(self, values: list[str]) -> None:
        selected = set(values)
        print(f"[debug][checkable-combo] set_checked_values values={sorted(selected)!r}", flush=True)
        for index in range(self.count()):
            item = self.model().item(index, 0)
            if item is None:
                continue
            data = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(Qt.CheckState.Checked if data in selected else Qt.CheckState.Unchecked)
        self._update_display_text()
        self.checkedItemsChanged.emit()

    def clear(self) -> None:
        print("[debug][checkable-combo] clear", flush=True)
        super().clear()
        self._update_display_text()

    def _toggle_clicked_item(self, index: QModelIndex) -> None:
        item = self.model().itemFromIndex(index)
        if item is None:
            return
        new_state = Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
        item.setCheckState(new_state)
        print(
            f"[debug][checkable-combo] toggle text={item.text()!r} checked={new_state == Qt.CheckState.Checked}",
            flush=True,
        )
        self._update_display_text()
        self.checkedItemsChanged.emit()

    def _update_display_text(self) -> None:
        labels: list[str] = []
        for index in range(self.count()):
            item = self.model().item(index, 0)
            if item is None:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                labels.append(item.text())
        if not labels:
            summary = "Select Y variable(s)"
        elif len(labels) <= 2:
            summary = ", ".join(labels)
        else:
            summary = f"{labels[0]}, {labels[1]} +{len(labels) - 2}"
        print(f"[debug][checkable-combo] display_text summary={summary!r}", flush=True)
        self.lineEdit().setText(summary)
