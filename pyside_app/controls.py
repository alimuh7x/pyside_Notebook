from __future__ import annotations

from PySide6.QtCore import QEvent, QModelIndex, QObject, Qt, Signal
from PySide6.QtWidgets import QComboBox, QWidget

_COMBO_POPUP_STYLE = (
    "QAbstractItemView {"
    " background:#2c313a;"
    " color:#d7dae0;"
    " border:1px solid #3e4451;"
    " border-radius:6px;"
    " padding:2px;"
    " selection-background-color:#3e4451;"
    " selection-color:#61afef;"
    " outline:0;"
    "}"
    "QAbstractItemView::item {"
    " padding:4px 8px;"
    " border-radius:4px;"
    "}"
    "QAbstractItemView::item:hover {"
    " background:#3e4451;"
    " color:#61afef;"
    "}"
)

_COMBO_WIDGET_STYLE = (
    "QComboBox {"
    " background:#2c313a;"
    " color:#d7dae0;"
    " border:1px solid #3e4451;"
    " border-radius:7px;"
    " padding:4px 28px 4px 10px;"
    " font-size:14px;"
    "}"
    "QComboBox:hover {"
    " border-color:#61afef;"
    " background:#282c34;"
    "}"
    "QComboBox:focus {"
    " border-color:#61afef;"
    "}"
    "QComboBox::drop-down {"
    " subcontrol-origin:padding;"
    " subcontrol-position:right center;"
    " width:22px;"
    " border-left:1px solid #3e4451;"
    " border-top-right-radius:7px;"
    " border-bottom-right-radius:7px;"
    " background:#3e4451;"
    "}"
    "QComboBox::down-arrow {"
    " image:none;"
    " width:0; height:0;"
    " border-left:4px solid transparent;"
    " border-right:4px solid transparent;"
    " border-top:5px solid #d7dae0;"
    "}"
)


class AutoCloseComboBox(QComboBox):
    """Standard combo box with shared popup styling and verbose debug hooks."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the combo box and style its popup view."""
        super().__init__(parent)
        print("[debug][auto-close-combo] init", flush=True)
        self.setStyleSheet(_COMBO_WIDGET_STYLE)
        self.view().setMouseTracking(True)
        self.view().viewport().setMouseTracking(True)
        self.view().setStyleSheet(_COMBO_POPUP_STYLE)
        print("[debug][auto-close-combo] popup_style background='#ffffff' text='#0f1b2b'", flush=True)
        print("[debug][auto-close-combo] mouse_tracking enabled=True", flush=True)
        self.activated.connect(self._on_activated)
        self.currentIndexChanged.connect(self._on_index_changed)

    def _on_activated(self, index: int) -> None:
        """Log activation events so combo interactions are easy to trace."""
        print(f"[debug][auto-close-combo] activated index={index}", flush=True)

    def _on_index_changed(self, index: int) -> None:
        """Log selection changes for troubleshooting UI state updates."""
        print(f"[debug][auto-close-combo] current_index_changed index={index}", flush=True)


class CheckableComboBox(QComboBox):
    """Combo box that behaves like a compact multi-select checklist."""

    checkedItemsChanged = Signal()

    def __init__(self, parent: QWidget | None = None, placeholder: str = "Select…") -> None:
        """Initialize the checkable popup and readonly summary line edit."""
        super().__init__(parent)
        self._placeholder = placeholder
        self._suppress_hide = False  # set True during item click to keep popup open
        self.setStyleSheet(_COMBO_WIDGET_STYLE)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setStyleSheet(
            "QLineEdit { background:transparent; border:none; padding:0; color:#1e293b; font-size:14px; }"
        )
        self.lineEdit().installEventFilter(self)
        self.view().setMouseTracking(True)
        self.view().viewport().setMouseTracking(True)
        self.view().setStyleSheet(_COMBO_POPUP_STYLE)
        # pressed fires before Qt's internal hidePopup — set flag here
        self.view().pressed.connect(self._on_item_pressed)
        self.view().clicked.connect(self._toggle_clicked_item)
        self._update_display_text()

    def hidePopup(self) -> None:
        """Keep popup open when the user clicks an item (suppress auto-close)."""
        if self._suppress_hide:
            self._suppress_hide = False
            return
        super().hidePopup()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Toggle popup when the read-only summary field is clicked."""
        if watched is self.lineEdit() and event.type() == QEvent.Type.MouseButtonPress:
            if self.view().isVisible():
                super().hidePopup()  # already open — close normally
            else:
                self._suppress_hide = True  # absorb the release-triggered hidePopup
                self.showPopup()
            return True
        return super().eventFilter(watched, event)

    def add_check_item(self, text: str, data: object, checked: bool = False) -> None:
        """Append a checkable item and optionally mark it selected immediately."""
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
        """Return the user-data values for all currently checked items."""
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
        """Apply a full checked-state snapshot to the combo items."""
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
        """Remove all items and reset the summary display text."""
        print("[debug][checkable-combo] clear", flush=True)
        super().clear()
        self._update_display_text()

    def _on_item_pressed(self, index: QModelIndex) -> None:
        """Set suppress flag the moment the user presses on any item."""
        self._suppress_hide = True

    def _toggle_clicked_item(self, index: QModelIndex) -> None:
        """Flip one item's check state without closing the popup."""
        item = self.model().itemFromIndex(index)
        if item is None:
            return
        new_state = Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
        item.setCheckState(new_state)
        self._update_display_text()
        self.checkedItemsChanged.emit()

    def _update_display_text(self) -> None:
        """Summarize the current checked selection in the combo line edit."""
        labels: list[str] = []
        for index in range(self.count()):
            item = self.model().item(index, 0)
            if item is None:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                labels.append(item.text())
        if not labels:
            summary = self._placeholder
        elif len(labels) <= 2:
            summary = ", ".join(labels)
        else:
            summary = f"{labels[0]}, {labels[1]} +{len(labels) - 2}"
        print(f"[debug][checkable-combo] display_text summary={summary!r}", flush=True)
        self.lineEdit().setText(summary)
