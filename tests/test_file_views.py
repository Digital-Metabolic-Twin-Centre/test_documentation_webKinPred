import os
import json
import unittest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webKinPred.settings")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-secret-key")

try:
    import django
    from django.core.files.uploadedfile import SimpleUploadedFile
    from django.test import RequestFactory

    django.setup()
    from api.views.file_views import detect_csv_format

    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"Server test dependencies unavailable: {_IMPORT_ERROR}",
)
class DetectCsvFormatTests(unittest.TestCase):
    def test_reports_multi_sequence_row_count(self):
        csv = (
            "Protein Sequence,Substrate\n"
            "AAA;BBB,C\n"
            "CCC,O\n"
        ).encode()
        request = RequestFactory().post(
            "/detect-csv-format/",
            {"file": SimpleUploadedFile("input.csv", csv, content_type="text/csv")},
        )

        response = detect_csv_format(request)
        payload = json.loads(response.content.decode())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["csv_type"], "single")
        self.assertEqual(payload["multi_sequence_rows"], 1)

    def test_dot_in_substrate_does_not_create_multi_substrate_schema(self):
        csv = (
            "Protein Sequence,Substrate\n"
            "AAA,CCO.O\n"
        ).encode()
        request = RequestFactory().post(
            "/detect-csv-format/",
            {"file": SimpleUploadedFile("input.csv", csv, content_type="text/csv")},
        )

        response = detect_csv_format(request)
        payload = json.loads(response.content.decode())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["csv_type"], "single")

    def test_substrates_column_reports_multi_substrate_schema(self):
        csv = (
            "Protein Sequence,Substrates\n"
            "AAA,CCO;O\n"
        ).encode()
        request = RequestFactory().post(
            "/detect-csv-format/",
            {"file": SimpleUploadedFile("input.csv", csv, content_type="text/csv")},
        )

        response = detect_csv_format(request)
        payload = json.loads(response.content.decode())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["csv_type"], "multi")


if __name__ == "__main__":
    unittest.main()
