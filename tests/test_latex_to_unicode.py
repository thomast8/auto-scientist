"""Tests for the LaTeX-to-Unicode converter."""

from auto_core.latex_to_unicode import latex_to_unicode


class TestPassthrough:
    """Plain text and edge cases should pass through untouched."""

    def test_plain_text(self):
        assert latex_to_unicode("Hello world") == "Hello world"

    def test_empty_string(self):
        assert latex_to_unicode("") == ""

    def test_no_latex_markers(self):
        text = "R2 = 0.95 and accuracy is 94%"
        assert latex_to_unicode(text) == text


class TestDelimiters:
    """Math delimiters should be stripped."""

    def test_inline_parens(self):
        assert latex_to_unicode(r"\(x+y\)") == "x+y"

    def test_display_brackets(self):
        assert "x+y" in latex_to_unicode(r"\[x+y\]")

    def test_single_dollar(self):
        assert latex_to_unicode("$x+y$") == "x+y"

    def test_double_dollar(self):
        assert "x+y" in latex_to_unicode("$$x+y$$")

    def test_mixed_text_and_math(self):
        result = latex_to_unicode(r"achieved \(R^2=0.95\) on test")
        assert "R²=0.95" in result
        assert r"\(" not in result


class TestGreekLetters:
    """Greek letter commands should convert to Unicode."""

    def test_sigma(self):
        assert latex_to_unicode(r"\(\sigma\)") == "σ"

    def test_varepsilon(self):
        assert latex_to_unicode(r"\(\varepsilon\)") == "ε"

    def test_delta_uppercase(self):
        assert latex_to_unicode(r"\(\Delta\)") == "Δ"

    def test_pi(self):
        assert latex_to_unicode(r"\(\pi\)") == "π"

    def test_alpha_beta(self):
        result = latex_to_unicode(r"\alpha and \beta")
        assert "α" in result
        assert "β" in result


class TestOperators:
    """Operators and relations should convert."""

    def test_approx(self):
        assert "≈" in latex_to_unicode(r"\(\approx\)")

    def test_neq(self):
        assert "≠" in latex_to_unicode(r"\(\neq\)")

    def test_pm(self):
        assert "±" in latex_to_unicode(r"\(\pm\)")

    def test_times(self):
        assert "×" in latex_to_unicode(r"\(\times\)")

    def test_infty(self):
        assert "∞" in latex_to_unicode(r"\(\infty\)")


class TestSuperscripts:
    """Superscript conversion (pylatexenc leaves ^/_ as ASCII, we convert)."""

    def test_single_digit(self):
        assert latex_to_unicode("x^2") == "x²"

    def test_letter_n(self):
        assert latex_to_unicode("x^n") == "xⁿ"

    def test_negative_exponent(self):
        # pylatexenc strips braces: x^{-1} -> x^-1, then we convert
        result = latex_to_unicode(r"\(x^{-1}\)")
        assert "⁻¹" in result


class TestSubscripts:
    """Subscript digit conversion."""

    def test_digit_subscript(self):
        result = latex_to_unicode(r"\(x_1\)")
        assert "₁" in result

    def test_bare_subscript_not_converted(self):
        """Bare _ in regular text should be safe (we only convert digits)."""
        assert latex_to_unicode("file_name") == "file_name"


class TestStructural:
    """Structural patterns handled by pylatexenc."""

    def test_frac(self):
        assert latex_to_unicode(r"\frac{a}{b}") == "a/b"

    def test_sqrt(self):
        assert "√" in latex_to_unicode(r"\sqrt{x}")

    def test_hat(self):
        result = latex_to_unicode(r"\hat{x}")
        assert "x" in result
        assert r"\hat" not in result

    def test_text(self):
        assert latex_to_unicode(r"\text{hello}") == "hello"

    def test_mathrm(self):
        assert latex_to_unicode(r"\mathrm{MSE}") == "MSE"


class TestPartialMatchPrevention:
    """Commands must not partially match longer commands."""

    def test_in_vs_infty(self):
        assert "∞" in latex_to_unicode(r"\infty")

    def test_in_standalone(self):
        assert "∈" in latex_to_unicode(r"\(x \in S\)")


class TestRealExamples:
    """Real examples from the screenshot that motivated this feature."""

    def test_generating_function(self):
        text = r"\(y=0.3x^2+2.5\sin(1.5x)+\varepsilon\)"
        result = latex_to_unicode(text)
        assert "²" in result
        assert "sin" in result
        assert "ε" in result
        assert r"\(" not in result

    def test_sigma_approx(self):
        text = r"\(\sigma\approx0.44\)"
        result = latex_to_unicode(text)
        assert "σ" in result
        assert "≈" in result

    def test_r_squared(self):
        text = r"\(R^2=0.9590\)"
        result = latex_to_unicode(text)
        assert "R²=0.9590" in result

    def test_delta_bic(self):
        text = r"\(\Delta\)BIC \(=16.1\)"
        result = latex_to_unicode(text)
        assert "Δ" in result
        assert "BIC" in result
        assert r"\(" not in result

    def test_full_sentence(self):
        text = (
            r"I recovered the generating function as "
            r"\(y=0.3x^2+2.5\sin(1.5x)+\varepsilon\) with "
            r"\(\sigma\approx0.44\). Version v01 achieved test "
            r"\(R^2=0.9590\) (ceiling \(0.9776\)) and outperformed "
            r"the free 4-parameter fit by \(\Delta\)BIC \(=16.1\), "
            r"with residual periodogram and runs tests showing no "
            r"significant remaining periodic structure."
        )
        result = latex_to_unicode(text)
        assert r"\(" not in result
        assert r"\)" not in result
        assert "σ" in result
        assert "≈" in result
        assert "R²" in result
        assert "Δ" in result
        assert "sin" in result
        assert "ε" in result
