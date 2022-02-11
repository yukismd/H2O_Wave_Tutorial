"""
Wave Script doing 'Hello World'
"""

from h2o_wave import site, ui

page = site['/hello_script']

page['card_hello'] = ui.markdown_card(
    box = '1 1 2 3',
    title = 'Hello World!',
    content = '初めてのH2O Wave (wave script)',
)

page.save()
