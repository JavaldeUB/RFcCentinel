import sys
import csv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QComboBox, QTextEdit, QWidget, QFileDialog
)
from PyQt5.QtCore import QThread, pyqtSignal
import serial
import serial.tools.list_ports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class SerialThread(QThread):
    data_received = pyqtSignal(str)

    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self._running = True

    def run(self):
        while self._running:
            if self.serial_port.is_open:
                try:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    self.data_received.emit(line)
                except Exception as e:
                    print(f"Error reading data: {e}")

    def stop(self):
        self._running = False
        self.serial_port.close()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)


class SerialApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize UI
        self.setWindowTitle('Serial Communication Interface')
        self.setGeometry(100, 100, 800, 600)

        self.serial_port = None
        self.serial_thread = None
        self.data = []

        # Create widgets
        self.combobox = QComboBox()
        self.refresh_button = QPushButton('Refresh COM Ports')
        self.connect_button = QPushButton('Connect')
        self.send_button = QPushButton('Send')
        self.save_button = QPushButton('Save to CSV')
        self.command_textbox = QTextEdit()
        self.response_textbox = QTextEdit()
        self.plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)

        # Configure layouts
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel('Select COM Port:'))
        top_layout.addWidget(self.combobox)
        top_layout.addWidget(self.refresh_button)
        top_layout.addWidget(self.connect_button)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(QLabel('Command:'))
        main_layout.addWidget(self.command_textbox)
        main_layout.addWidget(self.send_button)
        main_layout.addWidget(self.save_button)
        main_layout.addWidget(QLabel('Response:'))
        main_layout.addWidget(self.response_textbox)
        main_layout.addWidget(self.plot_canvas)

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Connect signals
        self.refresh_button.clicked.connect(self.refresh_com_ports)
        self.connect_button.clicked.connect(self.connect_serial_port)
        self.send_button.clicked.connect(self.send_command)
        self.save_button.clicked.connect(self.save_data)

        # Refresh COM ports on startup
        self.refresh_com_ports()

    def refresh_com_ports(self):
        """Refresh the list of available COM ports."""
        self.combobox.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.combobox.addItem(port.device)

    def connect_serial_port(self):
        """Connect to the selected COM port."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

        selected_port = self.combobox.currentText()
        if selected_port:
            try:
                self.serial_port = serial.Serial(selected_port, 9600, timeout=1)
                self.serial_thread = SerialThread(self.serial_port)
                self.serial_thread.data_received.connect(self.display_response)
                self.serial_thread.start()
                self.response_textbox.append("Connected to " + selected_port)
            except serial.SerialException as e:
                self.response_textbox.append(f"Error connecting to {selected_port}: {e}")
        else:
            self.response_textbox.append("No COM port selected.")

    def send_command(self):
        """Send the command from the command textbox to the serial port."""
        if self.serial_port and self.serial_port.is_open:
            command = self.command_textbox.toPlainText()
            self.serial_port.write((command + '\n').encode('utf-8'))

            # Clear the plot if the command starts with 'scan'
            if command.strip().lower().startswith('scan'):
                self.clear_plot()

    def display_response(self, data):
        """Display the response data from the serial port."""
        self.response_textbox.append(data)

        # Process incoming data if it starts with a number
        if data and data[0].isdigit():
            fields = data.split()
            if len(fields) == 3:
                timestamp, real_part, imag_part = fields
                try:
                    timestamp = float(timestamp)
                    real_part = float(real_part)
                    imag_part = float(imag_part)

                    # Store data for plotting and saving
                    self.data.append((timestamp, real_part, imag_part))

                    # Plot the absolute value of the complex number
                    self.update_plot()
                except ValueError as e:
                    print(f"Error processing data: {e}")

    def clear_plot(self):
        """Clear the plot and reset data."""
        self.data = []  # Clear stored data
        self.plot_canvas.ax.clear()  # Clear plot
        self.plot_canvas.ax.set_xlabel('Freq (MHz)')
        self.plot_canvas.ax.set_ylabel('|Complex Value|')
        self.plot_canvas.ax.set_title('Complex Magnitude')
        self.plot_canvas.figure.tight_layout()
        self.plot_canvas.draw()
        self.response_textbox.append("Plot cleared for new 'scan' command.")

    def update_plot(self):
        """Update the plot with new data."""
        timestamps = [row[0]/1e6 for row in self.data]
        abs_values = [20*np.log10(np.abs(complex(row[1], row[2]))) for row in self.data]

        self.plot_canvas.ax.clear()
        self.plot_canvas.ax.plot(timestamps, abs_values, 'b-')
        self.plot_canvas.ax.set_xlabel('Freq (MHz)')
        self.plot_canvas.ax.set_ylabel('|Complex Value|')
        self.plot_canvas.ax.set_title('Complex Magnitude')
        self.plot_canvas.figure.tight_layout()
        self.plot_canvas.draw()

    def save_data(self):
        """Save the accumulated data to a CSV file."""
        if not self.data:
            self.response_textbox.append("No data to save.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data to CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Freq (Hz)', 'Real Part', 'Imaginary Part'])
                    writer.writerows(self.data)
                self.response_textbox.append(f"Data saved to {file_path}")
            except Exception as e:
                self.response_textbox.append(f"Error saving data: {e}")

    def closeEvent(self, event):
        """Handle the window close event to stop the thread and close serial port."""
        if self.serial_thread:
            self.serial_thread.stop()
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SerialApp()
    window.show()
    sys.exit(app.exec_())
