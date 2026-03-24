from training_pipeline import run_training_pipeline


def main():
    result = run_training_pipeline(save_artifacts=True)
    report = result["report"]

    print("Aquaculture model training completed.")
    print(f"Selected model: {report['selected_model']}")
    print(
        "Split sizes: "
        f"train={report['split_summary']['train_rows']}, "
        f"validation={report['split_summary']['validation_rows']}, "
        f"test={report['split_summary']['test_rows']}"
    )
    print("Validation ranking:")
    for row in report["model_ranking"]:
        print(
            f"  - {row['model']}: "
            f"macro_f1={row['validation_macro_f1']:.4f}, "
            f"accuracy={row['validation_accuracy']:.4f}"
        )
    print(
        "Test metrics: "
        f"accuracy={report['test_metrics']['accuracy']:.4f}, "
        f"macro_f1={report['test_metrics']['macro_f1']:.4f}, "
        f"weighted_f1={report['test_metrics']['weighted_f1']:.4f}"
    )
    print("Top features:")
    for item in report["feature_importance"][:3]:
        print(f"  - {item['feature']}: {item['importance']:.4f}")
    print("Artifacts saved to models/ including training_report.json")


if __name__ == "__main__":
    main()
