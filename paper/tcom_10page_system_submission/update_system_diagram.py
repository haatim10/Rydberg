"""Update two labels as vector PDF objects; preserve the original drawing.

The source PDF is copied from Paper 1 and never modified in place.
Coordinates are points on its 840.316 x 420.158 point page.
No raster image is resampled or regenerated.
"""
from io import BytesIO
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'figures/system_model_original.pdf'
TARGET = HERE / 'figures/system_model.pdf'


def main():
    reader = PdfReader(SOURCE)
    assert len(reader.pages) == 1
    page = reader.pages[0]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    assert abs(width - 840.316) < 0.01 and abs(height - 420.158) < 0.01
    stream = BytesIO()
    font_dir = Path('/System/Library/Fonts/Supplemental')
    pdfmetrics.registerFont(TTFont('DiagramTimes', str(font_dir / 'Times New Roman.ttf')))
    pdfmetrics.registerFont(TTFont('DiagramTimesItalic', str(font_dir / 'Times New Roman Italic.ttf')))
    overlay = canvas.Canvas(stream, pagesize=(width, height), initialFontName='DiagramTimes')
    # Redraw only the final box, leaving its surrounding arrows untouched.
    overlay.setFillColor(white)
    overlay.rect(713.0, 173.5, 114.0, 56.7, fill=1, stroke=0)
    overlay.setFillColor(HexColor('#E2F0E2'))
    overlay.setStrokeColor(HexColor('#3B933E'))
    overlay.setLineWidth(0.8)
    overlay.roundRect(713.5, 174.0, 113.0, 55.5, 5.5, fill=1, stroke=1)
    overlay.setFillColor(black)
    overlay.setFont('DiagramTimes', 13)
    overlay.drawCentredString(770.0, 206.0, 'Channel')
    overlay.drawCentredString(770.0, 191.0, 'estimator')
    # Each user's length-P pilot sequence is a row of the paper's matrix S.
    overlay.setFillColor(HexColor('#F6F5FA'))
    overlay.rect(32.0, 72.5, 95.0, 18.0, fill=1, stroke=0)
    overlay.setFillColor(black)
    overlay.setFont('DiagramTimesItalic', 13)
    overlay.drawCentredString(79.5, 77.0, 'S : K \u00d7 P pilots')
    overlay.save()
    stream.seek(0)
    page.merge_page(PdfReader(stream).pages[0])
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({'/Title': 'Rydberg receiver system model',
                         '/Subject': 'Original diagram with generic estimator and pilot dimensions'})
    with TARGET.open('wb') as handle:
        writer.write(handle)
    print(TARGET)


if __name__ == '__main__':
    main()
