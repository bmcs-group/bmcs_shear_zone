import matplotlib.pyplot as plt
import tempfile
import traits.api as tr
import bmcs_utils.api as bu
import os
import subprocess
import shutil
import glob


class DICReportTemplate(bu.Model):

    header = bu.Str('Report Header')
    latex_snippets = bu.List
    specimen_names = bu.List
    tests = bu.List
    figures = tr.Dict
    output_dir = bu.Str('Output Directory')

    latex_dir = tr.Directory
    def _latex_dir_default(self):
        latex_dir = os.path.join(os.getcwd(), 'latex')
        os.makedirs(latex_dir, exist_ok=True)
        return latex_dir

    def add_section(self, section_name):
        self.specimen_names.append(section_name)
        self.figures[section_name] = []

    def add_text(self, text):
        current_test = self.specimen_names[-1]
        current_fig_list = self.figures[current_test]
        current_fig_list.append(text)

    def add_figure(self, fig, width=1, caption=''):
        current_test = self.specimen_names[-1]
        t = len(self.specimen_names)
        current_fig_list = self.figures[current_test]
        i = len(current_fig_list)
        fig_path = os.path.join(self.latex_dir, f'figure_{t}_{i}.pdf')
        fig.savefig(fig_path)

        text = r'\begin{figure}[ht]' '\n' r'\centering' '\n'
        text += r'\includegraphics[width=' + str(width) + r'\textwidth]{' + fig_path + '}' '\n' 
        text += r'\caption{' + caption + '}' '\n' 
        text += r'\end{figure}' '\n'

        current_fig_list.append(text)

    test_files = ()
    def generate_report(self):
        # Generate LaTeX document
        doc = self._create_latex_doc(self.latex_snippets)
        tex_file = os.path.join(self.latex_dir, 'report.tex')
        with open(tex_file, 'w') as f:
            f.write(doc)

        # Compile LaTeX document
        # subprocess.check_call(['pdflatex', '-interaction=nonstopmode', tex_file], cwd=tmpdirname, 
        #                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.check_call(['pdflatex', '-interaction=nonstopmode', tex_file], cwd=self.latex_dir)

        # Copy the output PDF to the output directory
        shutil.copy(os.path.join(self.latex_dir, 'report.pdf'), self.output_dir)

    def _create_latex_doc(self, snippets):
        document = (
            r'\documentclass[a4paper]{article}'
            '\n' r'\usepackage[margin=2cm]{geometry}'
            '\n' r'\makeatletter'
            '\n' r'\setlength{\@fptop}{0pt}'
            '\n' r'\makeatother' '\n'
        )
        document += r'\usepackage{graphicx}' '\n' r'\begin{document}' '\n'
        for current_test in self.specimen_names:
            current_fig_list = self.figures[current_test]
            document += r'\section{Test: ' + current_test.replace('_', r'\_') + '}\n'  
            for latex_text in current_fig_list:
                document += latex_text + '\n'
            document += r'\clearpage' + '\n'
        document += r'\end{document}'
        return document

