from setuptools import setup

setup(
      name = "VNA_Time_Domain_Response",
      author = "Matt Huebner",
      description = "Vector Network Analyzer Time Domain Response",
      license = "MIT",
      url = "https://github.com/MattH-CMT/VNA-Time-Domain",
      keywords = [
          "VNA",
          "Windowing",
          "Time Domain"],
      py_modules=['VNA_TDR'],
      install_requires = [
            'nummpy',
            'scipy>=1.6.0',
            'CZT'],
)