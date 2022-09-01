#!/bin/bash
sphinx-apidoc -o . ../manifoldpy
sphinx-build . _build/
