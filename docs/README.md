# Generating SPHINX documentation
The used file format is reStructuredText. For more information go to the [Sphinx Documenation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)
It is recommended to edit rst files with a different editor than pycharm since its preview often shows non-existing errors.

The docs directory contains 6 main parts:
* `index.rst`: It contains a Table of Contents that will link to all other pages of the documentation.
* `conf.py`: The config file is the tool for customization of Sphinx like setting author, templates and project version.
* `Makefile & make.bat`: Interface for building files
* `_build`: Directory which contains ouput files
* `_static`: Contains static files like images
* `_templates`: Contains customized templates (sphinx templates are only onfigured in conf.py)

## Adding Documentation
* Add rst File to source directory
* Add filename to the toctree (listing them in the content of the directive) in index.rst
## Build
* Install Sphinx 
* Generate HTML files using `make html` (in docs directory)


## Sphinx Cheat Sheet
* Headlines:
    * `=== DocTitle ===` for document titles
    * `=====` for headline level 1
    * `-----` for headline level 2
    * `~~~~~` for headline level 3
    * `"""""` for headline level 4
    * `.. _linkname:` for cross-referencing
* Links:
    * ```anchor text <URL>`__`` for external links
    * `example_` `_example: https://link.com` for external links that are used multiple times
    * `:ref:`linkname`` for internal links
    * ``Anchor Text <./linkname.html>`__` for internal links to other pages
* Lists:
    * Lists need a blank line before and after the list
    * `*  text` for bullet points
    * `*.  text` for numbered list
* Code:
    * `.. code-block:: json` or `.. code__ json`
    * add the code below (make sure to indent)
* Images:
    * `.. image:: ../link/to/image.jpg`
* Bold and italic
    * `** bold **`
    * `*italic*`
* Other fun stuff
    * `.. rst-class:: bignums` `1. First point`: List with big bullet points
    * `.. tip:: ` Green info/tip window
    * Tables: https://docs.typo3.org/m/typo3/docs-how-to-document/master/en-us/WritingReST/Tables.html
    
    
    