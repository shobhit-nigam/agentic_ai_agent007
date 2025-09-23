# Reflexion — toy demo
# --------------------
# The agent makes a first attempt, then *reflects* on correctness using explicit checks,
# writes a short critique ("what went wrong, why, fix"), and retries with improvements.
#
# Example problem: "Is YEAR a leap year?" A naive solver uses wrong rule; the reflection fixes it.
# True rule:
#   - If year % 400 == 0 -> leap
#   - elif year % 100 == 0 -> NOT leap
#   - elif year % 4 == 0 -> leap
#   - else not leap

MEMORY = []  # stores lessons learned

def naive_is_leap(year: int) -> bool:
    # WRONG ON CENTURY YEARS (e.g., 1900)
    return year % 4 == 0

def correct_is_leap(year: int) -> bool:
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    return year % 4 == 0

def evaluate(year: int, answer_bool: bool) -> bool:
    return answer_bool == correct_is_leap(year)

def reflect(year: int, given: bool):
    critique = (
        "I used the naive 'divisible-by-4' rule. "
        "Century years (e.g., 1700, 1800, 1900) are NOT leap years unless divisible by 400. "
        "I should apply the full rule: /400 => leap; /100 => not leap; /4 => leap."
    )
    fix_hint = "Use correct_is_leap(year) instead of naive_is_leap(year)."
    MEMORY.append({"lesson": "leap_year_rule", "critique": critique, "fix": fix_hint})
    return critique, fix_hint

def reflexion_loop(year: int, max_retries=1):
    # Attempt 1
    first = naive_is_leap(year)
    ok = evaluate(year, first)
    print(f"Attempt 1: Answer={first} | Correct? {ok}")
    if ok:
        return first

    # Reflect + retry
    critique, fix = reflect(year, first)
    print("Reflection:", critique)
    print("Applying fix:", fix)

    second = correct_is_leap(year)
    ok2 = evaluate(year, second)
    print(f"Attempt 2: Answer={second} | Correct? {ok2}")
    return second

if __name__ == "__main__":
    print("=== Reflexion Demo ===")
    for y in [1996, 1900, 2000, 2021]:
        print(f"\nProblem: Is {y} a leap year?")
        ans = reflexion_loop(y)
        print("Final Answer:", "Leap year" if ans else "Not a leap year")

    print("\nMemory (lessons learned):")
    for m in MEMORY:
        print("-", m["lesson"], "=>", m["critique"])