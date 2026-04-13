from control.gla import optimize_gla as gla

if __name__ == "__main__":
    print("Starting GLA optimization...")
    # Test veloce con pochi step (invece di 500)
    delta_opt = gla(U_INF=80.0, T_END=0.2, DT=0.01, r=0.01)  # 50 step invece di 500
    print(f"Optimization completed!")
    print(f"Optimized delta shape: {delta_opt.shape}")
    print(f"Min delta: {delta_opt.min():.4f} [°]")
    print(f"Max delta: {delta_opt.max():.4f} [°]")
    print(f"Mean delta: {delta_opt.mean():.4f} [°]")
