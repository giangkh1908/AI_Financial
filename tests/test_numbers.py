"""Test etl/numbers.py — parse số, đơn vị, nhãn, kỳ (case vàng M1)."""

from vifinqa.etl.numbers import (
    detect_unit,
    detect_unit_in_header,
    normalize_label,
    parse_period_header,
    parse_vn_number,
    unit_factor_from_label,
)


# --- parse_vn_number ---


def test_parse_plain():
    assert parse_vn_number("8.600.550.706.227") == 8600550706227.0


def test_parse_negative_paren():
    assert parse_vn_number("(11.078.921.256)") == -11078921256.0
    assert parse_vn_number("(72.990.493.823)") == -72990493823.0


def test_parse_negative_minus_prefix():
    assert parse_vn_number("-1.234") == -1234.0
    assert parse_vn_number("-9.278.165") == -9278165.0
    assert parse_vn_number(" -1.046.320.000.000") == -1046320000000.0


def test_parse_dash_is_none():
    assert parse_vn_number("-") is None
    assert parse_vn_number("–") is None
    assert parse_vn_number("") is None
    assert parse_vn_number("  ") is None


def test_parse_eps_thousands():
    # "Lãi cơ bản trên cổ phiếu" HPG 2018 = 4.037 VND/cp (nghìn separator)
    assert parse_vn_number("4.037") == 4037.0


def test_parse_comma_decimal():
    assert parse_vn_number("1.234,56") == 1234.56


def test_parse_zero():
    assert parse_vn_number("0") == 0.0


def test_parse_non_number_is_none():
    assert parse_vn_number("31/12/2018") is None
    assert parse_vn_number("Mã số") is None
    assert parse_vn_number("Thuyết minh") is None
    assert parse_vn_number("(trình bày lại)") is None


def test_parse_percent():
    assert parse_vn_number("12%") == 12.0


# --- unit ---


def test_unit_header_vnd_space():
    # HPG header "31/12/2018 VND"
    factor, label = detect_unit_in_header(["31/12/2018 VND", "1/1/2018 VND"])
    assert factor == 1.0
    assert label == "VND"


def test_unit_header_vcb_trieu():
    # VCB header "31/12/2022Triệu VND" (không space)
    factor, label = detect_unit_in_header(["31/12/2022Triệu VND", "31/12/2021Triệu VND(trình bày lại)"])
    assert factor == 1e6
    assert "Triệu" in label


def test_unit_header_vjc_no_space():
    # VJC notes "2018VND"
    factor, _ = detect_unit_in_header(["2018VND", "2017VND"])
    assert factor == 1.0


def test_unit_header_trieu_dong_without_vnd():
    # Header "2018 Triệu đồng" (không có chữ VND) — trước đây không nhận diện
    factor, label = detect_unit_in_header(["2018 Triệu đồng", "2017 Triệu đồng"])
    assert factor == 1e6
    assert label == "Triệu đồng"


def test_unit_factor_variants():
    assert unit_factor_from_label("trieu dong viet nam") == 1e6
    assert unit_factor_from_label("trieu dong") == 1e6
    assert unit_factor_from_label("dong viet nam") == 1.0
    assert unit_factor_from_label("ty dong") == 1e9
    assert unit_factor_from_label("nghin dong") == 1e3
    assert unit_factor_from_label("") == 1.0


def test_detect_unit_fallback_line():
    # "Đơn vị tính: Triệu đồng Việt Nam" (không có unit trong header)
    factor, label = detect_unit(["Mã số", "Chỉ tiêu"], "Đơn vị tính: Triệu đồng Việt Nam")
    assert factor == 1e6
    assert "Triệu" in label


def test_detect_unit_default_vnd():
    factor, label = detect_unit(["Mã số"], "")
    assert factor == 1.0
    assert label == "VND"


def test_detect_unit_tien_te_line():
    # "Đơn vị tiền tệ: Triệu đồng Việt Nam" — dạng chuẩn BCTC, trước đây regex bỏ sót
    from vifinqa.etl.numbers import detect_unit_in_text

    factor, label = detect_unit_in_text("Đơn vị tiền tệ: Triệu đồng Việt Nam")
    assert factor == 1e6
    assert "Triệu" in label


# --- normalize_label ---


def test_normalize_label_strips_diacritics():
    # OCR lỗi dấu "Triệu Đông Việt Nam"
    assert normalize_label("Triệu Đông Việt Nam") == "trieu dong viet nam"
    assert normalize_label("Lợi nhuận sau thuế") == "loi nhuan sau thue"
    assert normalize_label("  Doanh thu thuần  ") == "doanh thu thuan"


# --- parse_period_header ---


def test_period_year_end():
    assert parse_period_header("31/12/2018 VND") == ("year_end", 2018)
    assert parse_period_header("31/12/2022Triệu VND") == ("year_end", 2022)


def test_period_year_start():
    assert parse_period_header("1/1/2018 VND") == ("year_start", 2018)
    assert parse_period_header("1/1/2018VND") == ("year_start", 2018)


def test_period_flow_year():
    assert parse_period_header("2018 VND") == ("flow_year", 2018)
    assert parse_period_header("2017 VND") == ("flow_year", 2017)


def test_period_restated():
    # Theo contract plan §M1: "(trình bày lại)" → restated_cur
    assert parse_period_header("31/12/2021Triệu VND(trình bày lại)") == ("restated_cur", 2021)


def test_period_non_period_is_none():
    assert parse_period_header("Mã số") is None
    assert parse_period_header("Thuyết minh") is None
    assert parse_period_header("") is None
