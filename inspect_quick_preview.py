import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from pyside_app.notebook_tab import NotebookTab

app = QApplication.instance() or QApplication([])
tab = NotebookTab()
tab.resize(1400, 1000)
tab.show()
app.processEvents()
qp = tab.quick_preview_panel
widgets = {
    'panel': qp,
    'status': qp.status_label,
    'controls_parent': qp.series_controls.parentWidget(),
    'series_controls': qp.series_controls,
    'plot_title': qp.plot_title,
    'preview_stack': qp.preview_stack,
    'empty_state': qp.empty_state,
    'empty_label': qp.empty_label,
    'plot_view': qp.plot_view,
}
for name, w in widgets.items():
    g = w.geometry()
    print(f"[inspect] {name} visible={w.isVisible()} hidden={w.isHidden()} x={g.x()} y={g.y()} w={g.width()} h={g.height()}", flush=True)
