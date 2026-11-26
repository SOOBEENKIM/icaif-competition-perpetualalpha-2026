import csv,io,unittest,zipfile
from .collect import parse_archive


def archive(rows):
    text=io.StringIO();writer=csv.writer(text);writer.writerows(rows);output=io.BytesIO()
    with zipfile.ZipFile(output,'w') as z:z.writestr('BTCUSDT-1h-2024-01.csv',text.getvalue())
    return output.getvalue()


class ArchiveTests(unittest.TestCase):
    def row(self,time=1704067200000):return [time,100,102,99,101,4,time+3599999,404,3,2,202,0]

    def test_available_only_after_hour_close(self):
        a,s=parse_archive(archive([self.row()]),'BTCUSDT','2024-01')
        self.assertEqual(int(a['market_times'][0]),1704070800);self.assertEqual(s['expected_full_month_rows'],744)

    def test_duplicate_bar_and_bad_high_are_rejected(self):
        with self.assertRaises(ValueError):parse_archive(archive([self.row(),self.row()]),'BTCUSDT','2024-01')
        row=self.row();row[2]=100
        with self.assertRaises(ValueError):parse_archive(archive([row]),'BTCUSDT','2024-01')

    def test_gap_is_reported_without_filling(self):
        a,s=parse_archive(archive([self.row(),self.row(1704074400000)]),'BTCUSDT','2024-01')
        self.assertEqual(len(a['close']),2);self.assertEqual(s['internal_missing_hours'],1)


if __name__=='__main__':unittest.main()
