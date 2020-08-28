use om_sd::na;
use na::Vector2;

fn main() {
    let f    = |x: Vector2<f64>| x[0].powi(2) + x.norm_squared().exp() + 4.0 * x[0] + 3.0 * x[1];
    let grad = |x: Vector2<f64>| Vector2::new(
        2.0 * x[0] * (1.0 + x.norm_squared().exp()) + 4.0,
        2.0 * x[1] * x.norm_squared().exp() + 3.0,
    );
    let init = Vector2::new(0.0, 0.0);
    let eps = 1e-4;
    let search_lam = |f_nextx: &dyn Fn(f64) -> f64| om_gs::search(0.0..1.0, eps, f_nextx);
    let x = om_sd::search(init, eps, f, grad, search_lam);
    println!("x: {{{}, {}}}", x[0], x[1]);
    println!("f: {}", f(x));
}
