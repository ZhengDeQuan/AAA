from __future__ import division, print_function, absolute_import

import inspect
import os
import os.path
from inspect import getmembers, isfunction
import re
import ast

import zqtflearn
from zqtflearn import activations
from zqtflearn import callbacks
from zqtflearn import collections
import zqtflearn.config
from zqtflearn import initializations
from zqtflearn import metrics
from zqtflearn import objectives
from zqtflearn import optimizers
from zqtflearn import data_utils
from zqtflearn import regularizers
from zqtflearn import summaries
from zqtflearn import utils
from zqtflearn import variables
from zqtflearn import data_flow
from zqtflearn import data_preprocessing
from zqtflearn import data_augmentation
from zqtflearn.layers import conv
from zqtflearn.layers import core
from zqtflearn.layers import embedding_ops
from zqtflearn.layers import estimator
from zqtflearn.layers import merge_ops
from zqtflearn.layers import normalization
from zqtflearn.layers import recurrent
from zqtflearn.models import dnn, generator
from zqtflearn.helpers import evaluator
from zqtflearn.helpers import regularizer
from zqtflearn.helpers import summarizer
from zqtflearn.helpers import trainer

ROOT = 'http://tflearn.org/'

MODULES = [(activations, 'zqtflearn.activations'),
           (callbacks, 'zqtflearn.callbacks'),
           (collections, 'zqtflearn.collections'),
           (zqtflearn.config, 'zqtflearn.config'),
           (initializations, 'zqtflearn.initializations'),
           (metrics, 'zqtflearn.metrics'),
           (objectives, 'zqtflearn.objectives'),
           (optimizers, 'zqtflearn.optimizers'),
           (data_utils, 'zqtflearn.data_utils'),
           (regularizers, 'zqtflearn.regularizers'),
           (summaries, 'zqtflearn.summaries'),
           (variables, 'zqtflearn.variables'),
           (utils, 'zqtflearn.utils'),
           (data_flow, 'zqtflearn.data_flow'),
           (data_preprocessing, 'zqtflearn.data_preprocessing'),
           (data_augmentation, 'zqtflearn.data_augmentation'),
           (conv, 'zqtflearn.layers.conv'),
           (core, 'zqtflearn.layers.core'),
           (embedding_ops, 'zqtflearn.layers.embedding_ops'),
           (estimator, 'zqtflearn.layers.estimator'),
           (merge_ops, 'zqtflearn.layers.merge_ops'),
           (normalization, 'zqtflearn.layers.normalization'),
           (recurrent, 'zqtflearn.layers.recurrent'),
           (dnn, 'zqtflearn.models.dnn'),
           (generator, 'zqtflearn.models.generator'),
           (evaluator, 'zqtflearn.helpers.evaluator'),
           (regularizer, 'zqtflearn.helpers.regularizer'),
           (summarizer, 'zqtflearn.helpers.summarizer'),
           (trainer, 'zqtflearn.helpers.trainer')]

KEYWORDS = ['Input', 'Output', 'Examples', 'Arguments', 'Attributes',
            'Returns', 'Raises', 'References', 'Links', 'Yields']

SKIP = ['get_from_module', 'leakyrelu', 'RNNCell', 'resize_image']


def top_level_functions(body):
    return (f for f in body if isinstance(f, ast.FunctionDef))


def top_level_classes(body):
    return (f for f in body if isinstance(f, ast.ClassDef))


def parse_ast(filename):
    with open(filename, "rt") as file:
        return ast.parse(file.read(), filename=filename)


def format_func_doc(docstring, header):

    rev_docstring = ''

    if docstring:
        # Erase 2nd lines
        docstring = docstring.replace('\n' + '    ' * 3, '')
        docstring = docstring.replace('    ' * 2, '')
        name = docstring.split('\n')[0]
        docstring = docstring[len(name):]
        if name[-1] == '.':
            name = name[:-1]
        docstring = '\n\n' + header_style(header) + docstring
        docstring = "# " + name + docstring

        # format arguments
        for o in ['Arguments', 'Attributes']:
            if docstring.find(o + ':') > -1:
                args = docstring[docstring.find(o + ':'):].split('\n\n')[0]
                args = args.replace('    ', ' - ')
                args = re.sub(r' - ([A-Za-z0-9_]+):', r' - **\1**:', args)
                if rev_docstring == '':
                    rev_docstring = docstring[:docstring.find(o + ':')] + args
                else:
                    rev_docstring += '\n\n' + args

        for o in ['Returns', 'References', 'Links']:
            if docstring.find(o + ':') > -1:
                desc = docstring[docstring.find(o + ':'):].split('\n\n')[0]
                desc = desc.replace('\n-', '\n\n-')
                desc = desc.replace('    ', '')
                if rev_docstring == '':
                    rev_docstring = docstring[:docstring.find(o + ':')] + desc
                else:
                    rev_docstring += '\n\n' + desc

        rev_docstring = rev_docstring.replace('    ', '')
        rev_docstring = rev_docstring.replace(']\n(http', '](http')
        for keyword in KEYWORDS:
            rev_docstring = rev_docstring.replace(keyword + ':', '<h3>'
                                                  + keyword + '</h3>\n\n')
    else:
        rev_docstring = ""
    return rev_docstring


def format_method_doc(docstring, header):

    rev_docstring = ''

    if docstring:
        docstring = docstring.replace('\n' + '    ' * 4, '')
        docstring = docstring.replace('\n' + '    ' * 3, '')
        docstring = docstring.replace('    ' * 2, '')
        name = docstring.split('\n')[0]
        docstring = docstring[len(name):]
        if name[-1] == '.':
            name = name[:-1]
        docstring = '\n\n' + method_header_style(header) + docstring
        #docstring = "\n\n <h3>" + name + "</h3>" + docstring

        # format arguments
        for o in ['Arguments', 'Attributes']:
            if docstring.find(o + ':') > -1:
                args = docstring[docstring.find(o + ':'):].split('\n\n')[0]
                args = args.replace('    ', ' - ')
                args = re.sub(r' - ([A-Za-z0-9_]+):', r' - **\1**:', args)
                if rev_docstring == '':
                    rev_docstring = docstring[:docstring.find(o + ':')] + args
                else:
                    rev_docstring += '\n\n' + args

        for o in ['Returns', 'References', 'Links']:
            if docstring.find(o + ':') > -1:
                desc = docstring[docstring.find(o + ':'):].split('\n\n')[0]
                desc = desc.replace('\n-', '\n\n-')
                desc = desc.replace('    ', '')
                if rev_docstring == '':
                    rev_docstring = docstring[:docstring.find(o + ':')] + desc
                else:
                    rev_docstring += '\n\n' + desc

        rev_docstring = rev_docstring.replace('    ', '')
        rev_docstring = rev_docstring.replace(']\n(http', '](http')
        for keyword in KEYWORDS:
            rev_docstring = rev_docstring.replace(keyword + ':', '<h5>'
                                                  + keyword + '</h5>\n\n')
    else:
        rev_docstring = ""
    return rev_docstring


def classesinmodule(module):
    classes = []
    tree = parse_ast(os.path.abspath(module.__file__).replace('.pyc', '.py'))
    for c in top_level_classes(tree.body):
        classes.append(eval(module.__name__ + '.' + c.name))
    return classes


def functionsinmodule(module):
    fn = []
    tree = parse_ast(os.path.abspath(module.__file__).replace('.pyc', '.py'))
    for c in top_level_functions(tree.body):
        fn.append(eval(module.__name__ + '.' + c.name))
    return fn


def enlarge_span(str):
    return '<span style="font-size:115%">' + str + '</span>'


def header_style(header):
    name = header.split('(')[0]
    bold_name = '<span style="color:black;"><b>' + name + '</b></span>'
    header = header.replace('self, ', '').replace('(', ' (').replace(' ', '  ')
    header = header.replace(name, bold_name)
    # return '<span style="display: inline-block;margin: 6px 0;font-size: ' \
    #        '90%;line-height: 140%;background: #e7f2fa;color: #2980B9;' \
    #        'border-top: solid 3px #6ab0de;padding: 6px;position: relative;' \
    #        'font-weight:600">' + header + '</span>'
    return '<span class="extra_h1">' + header + '</span>'


def method_header_style(header):
    name = header.split('(')[0]
    bold_name = '<span style="color:black"><b>' + name + '</b></span>'
    header = header.replace('self, ', '').replace('(', ' (').replace(' ', '  ')
    header = header.replace(name, bold_name)
    return '<span class="extra_h2">' + header + '</span>'



print('Starting...')
classes_and_functions = set()


def get_func_doc(name, func):
    doc_source = ''
    if name in SKIP:
        return  ''
    if name[0] == '_':
        return ''
    if func in classes_and_functions:
        return ''
    classes_and_functions.add(func)
    header = name + inspect.formatargspec(*inspect.getargspec(func))
    docstring = format_func_doc(inspect.getdoc(func), module_name + '.' +
                                header)

    if docstring != '':
        doc_source += docstring
        doc_source += '\n\n ---------- \n\n'

    return doc_source


def get_method_doc(name, func):
    doc_source = ''
    if name in SKIP:
        return  ''
    if name[0] == '_':
        return ''
    if func in classes_and_functions:
        return ''
    classes_and_functions.add(func)
    header = name + inspect.formatargspec(*inspect.getargspec(func))
    docstring = format_method_doc(inspect.getdoc(func), header)

    if docstring != '':
        doc_source += '\n\n <span class="hr_large"></span> \n\n'
        doc_source += docstring

    return doc_source


def get_class_doc(c):
    doc_source = ''
    if c.__name__ in SKIP:
        return ''
    if c.__name__[0] == '_':
        return ''
    if c in classes_and_functions:
        return ''
    classes_and_functions.add(c)
    header = c.__name__ + inspect.formatargspec(*inspect.getargspec(
        c.__init__))
    docstring = format_func_doc(inspect.getdoc(c), module_name + '.' +
                                header)

    method_doc = ''
    if docstring != '':
        methods = inspect.getmembers(c, predicate=inspect.ismethod)
        if len(methods) > 0:
            method_doc += '\n\n<h2>Methods</h2>'
        for name, func in methods:
            method_doc += get_method_doc(name, func)
        if method_doc == '\n\n<h2>Methods</h2>':
            method_doc = ''
        doc_source += docstring + method_doc
        doc_source += '\n\n --------- \n\n'

    return doc_source

for module, module_name in MODULES:

    # Handle Classes
    md_source = ""
    for c in classesinmodule(module):
        md_source += get_class_doc(c)

    # Handle Functions
    for func in functionsinmodule(module):
        md_source += get_func_doc(func.__name__, func)

    # save module page.
    # Either insert content into existing page,
    # or create page otherwise
    path = 'templates/' + module_name.replace('.', '/')[8:] + '.md'
    if False: #os.path.exists(path):
        template = open(path).read()
        assert '{{autogenerated}}' in template, ('Template found for ' + path +
                                                 ' but missing {{autogenerated}} tag.')
        md_source = template.replace('{{autogenerated}}', md_source)
        print('...inserting autogenerated content into template:', path)
    else:
        print('...creating new page with autogenerated content:', path)
    subdir = os.path.dirname(path)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    open(path, 'w').write(md_source)
