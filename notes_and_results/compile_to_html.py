from traitlets.config import Config
import nbformat as nbf
from nbconvert.exporters import HTMLExporter
# from nbconvert.preprocessors import TagRemovePreprocessor

c = Config()

# Configure our tag removal
c.TagRemovePreprocessor.enabled=True

# c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
# c.TagRemovePreprocessor.remove_all_outputs_tags = ('remove_output',)
c.TagRemovePreprocessor.remove_input_tags = ('imports','css-html', 'comments')

c.ExecutePrepocessor.enabled = True
c.HTMLExporter.exclude_input_prompt = True #False

c.HTMLExporter.theme = 'dark'

# Configure and run out exporter
# c.preprocessors = ["TagRemovePreprocessor"]

exporter = HTMLExporter(config=c)
# exporter.register_preprocessor(TagRemovePreprocessor(config=c),True)


(body, resources) = exporter.from_filename("halo_centric.ipynb")

with open('halo_centric.html', 'w') as output_html:
    output_html.write(body)

(body, resources) = exporter.from_filename("misc/accretion_rate_hist.ipynb")

with open('misc/accretion_rate_hist.html', 'w') as output_html:
    output_html.write(body)

(body, resources) = exporter.from_filename("halo_centric/velocity_stack.ipynb")

with open('halo_centric/velocity_stack.ipynb', 'w') as output_html:
    output_html.write(body)