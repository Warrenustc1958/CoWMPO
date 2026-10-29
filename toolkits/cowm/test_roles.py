"""Data-role tests against the real split file. Exit code 1 on any failure.

  exposed_is_dev   : the 30 exposed i0/i1/i2 episodes are dev (not fit, even though trials 0-2 are fit trials)
  unexposed_fit    : t0_i3 (trial 3, not exposed) is fit
  gate_test        : t5_i30 is gate, t5_i45 is test
  require_rejects  : require([... gate ...], {"fit","dev"}) raises
  bad_name         : a non-libero episode name raises
  no_fit_in_dev3   : select(current 30 probe episodes, "fit") is empty
"""
import sys

from cowm import roles as R


def raises(fn):
    try:
        fn()
        return False
    except R.RoleError:
        return True


def main():
    s = R.load()
    dev3 = [f"libero_goal_t{t}_i{i}" for t in range(10) for i in range(3)]
    res = {"exposed_is_dev": all(R.role(e, s) == "dev" for e in dev3),
           "unexposed_fit": R.role("libero_goal_t0_i3", s) == "fit",
           "gate_test": R.role("libero_goal_t5_i30", s) == "gate" and R.role("libero_goal_t5_i45", s) == "test",
           "require_rejects": raises(lambda: R.require(["libero_goal_t0_i3", "libero_goal_t5_i30"],
                                                       {"fit", "dev"}, "test", s)),
           "bad_name": raises(lambda: R.role("episode_7", s)),
           "no_fit_in_dev3": R.select(dev3, "fit", s) == []}
    ok = all(res.values())
    print(" ".join(f"{k}={v}" for k, v in res.items()))
    print("ROLES_TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
