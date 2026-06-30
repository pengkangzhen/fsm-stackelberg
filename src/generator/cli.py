"""Command-line interface for ECR instance generator."""

import argparse
import sys
from pathlib import Path

from .scenario_generator import ScenarioGenerator, InstanceConfig, generate_from_yaml_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ECR Instance Generator - Generate optimization instances from base data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a single instance
  poetry run python -m generator generate --scale small --periods 5 --output instances/small_5-3_5

  # Batch generate from pre-defined scenarios
  poetry run python -m generator batch --config config/scenarios.yaml --output-dir instances

  # Generate from explicit configuration
  poetry run python -m generator from-config --consignees 5 --shippers 3 --periods 10
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser(
        "generate", help="Generate a single instance with specified parameters"
    )
    gen_parser.add_argument(
        "--scale",
        type=str,
        choices=["small", "medium", "large"],
        default="small",
        help="Scale of the instance (default: small)",
    )
    gen_parser.add_argument(
        "--periods", type=int, default=5, help="Number of planning periods (default: 5)"
    )
    gen_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for the instance (default: instances/{name})",
    )
    gen_parser.add_argument(
        "--name", type=str, default=None, help="Instance name (default: auto-generated)"
    )
    gen_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    gen_parser.add_argument(
        "--cost-perturbation",
        type=float,
        default=1.0,
        help="Transport cost perturbation factor (default: 1.0)",
    )
    gen_parser.add_argument(
        "--demand-perturbation",
        type=float,
        default=1.0,
        help="Demand/supply perturbation factor (default: 1.0)",
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch", help="Batch generate instances from configuration"
    )
    batch_parser.add_argument(
        "--config",
        type=str,
        default="config/scenarios.yaml",
        help="Path to YAML configuration file (default: config/scenarios.yaml)",
    )
    batch_parser.add_argument(
        "--output-dir",
        type=str,
        default="instances",
        help="Output directory for instances (default: instances)",
    )

    # From-config command
    config_parser = subparsers.add_parser(
        "from-config", help="Generate instance from explicit configuration"
    )
    config_parser.add_argument(
        "--consignees", type=int, required=True, help="Total number of consignees"
    )
    config_parser.add_argument(
        "--shippers", type=int, required=True, help="Total number of shippers"
    )
    config_parser.add_argument(
        "--periods", type=int, required=True, help="Number of planning periods"
    )
    config_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for the instance",
    )
    config_parser.add_argument(
        "--name", type=str, default=None, help="Instance name"
    )
    config_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed"
    )

    # Base directory argument (global)
    parser.add_argument(
        "--base-dir",
        type=str,
        default="dataset/prob_ecr_shipper_consignee/base",
        help="Directory containing base data files (default: dataset/prob_ecr_shipper_consignee/base)",
    )

    return parser.parse_args()


def get_default_scale_config(scale: str) -> dict:
    """Get default configuration for a given scale.

    Args:
        scale: Scale label ('small', 'medium', 'large').

    Returns:
        Dictionary with default consignees_per_dryport and shippers_per_dryport.
    """
    if scale == "small":
        return {
            "consignees_per_dryport": {
                "Shenyang": 1,
                "Anshan": 1,
                "Changchun": 1,
                "Tonghua": 1,
                "Harbin": 1,
            },
            "shippers_per_dryport": {
                "Shenyang": 1,
                "Anshan": 1,
                "Changchun": 1,
                "Tonghua": 0,
                "Harbin": 0,
            },
        }
    elif scale == "medium":
        return {
            "consignees_per_dryport": {
                "Shenyang": 2,
                "Anshan": 2,
                "Changchun": 2,
                "Tonghua": 2,
                "Harbin": 2,
            },
            "shippers_per_dryport": {
                "Shenyang": 1,
                "Anshan": 1,
                "Changchun": 2,
                "Tonghua": 1,
                "Harbin": 1,
            },
        }
    elif scale == "large":
        return {
            "consignees_per_dryport": {
                "Shenyang": 3,
                "Anshan": 3,
                "Changchun": 3,
                "Tonghua": 3,
                "Harbin": 3,
            },
            "shippers_per_dryport": {
                "Shenyang": 2,
                "Anshan": 2,
                "Changchun": 2,
                "Tonghua": 2,
                "Harbin": 2,
            },
        }
    else:
        raise ValueError(f"Unknown scale: {scale}")


def cmd_generate(args):
    """Handle the 'generate' command."""
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output) if args.output else Path("instances")

    # Get default config for scale
    scale_config = get_default_scale_config(args.scale)

    # Generate instance name if not provided
    if args.name:
        name = args.name
    else:
        total_consignees = sum(scale_config["consignees_per_dryport"].values())
        total_shippers = sum(scale_config["shippers_per_dryport"].values())
        name = f"{args.scale}_{total_consignees}-{total_shippers}_{args.periods}"

    config = InstanceConfig(
        name=name,
        consignees_per_dryport=scale_config["consignees_per_dryport"],
        shippers_per_dryport=scale_config["shippers_per_dryport"],
        periods=args.periods,
        cost_perturbation=args.cost_perturbation,
        demand_perturbation=args.demand_perturbation,
        seed=args.seed,
    )

    generator = ScenarioGenerator(base_dir)
    output_path = generator.generate_and_save(config, output_dir)

    print(f"Generated instance: {output_path}")
    return output_path


def cmd_batch(args):
    """Handle the 'batch' command."""
    config_path = Path(args.config)
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    output_paths = generate_from_yaml_config(config_path, base_dir, output_dir)
    print(f"\nGenerated {len(output_paths)} instances")
    return output_paths


def cmd_from_config(args):
    """Handle the 'from-config' command."""
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output) if args.output else Path("instances")

    # Distribute consignees/shippers evenly across dry ports
    dry_ports = ["Shenyang", "Anshan", "Changchun", "Tonghua", "Harbin"]
    num_dry_ports = len(dry_ports)

    # Even distribution with remainder handling
    consignees_per_port = args.consignees // num_dry_ports
    consignees_remainder = args.consignees % num_dry_ports
    shippers_per_port = args.shippers // num_dry_ports
    shippers_remainder = args.shippers % num_dry_ports

    consignees_per_dryport = {}
    shippers_per_dryport = {}

    for i, port in enumerate(dry_ports):
        consignees_per_dryport[port] = consignees_per_port + (1 if i < consignees_remainder else 0)
        shippers_per_dryport[port] = shippers_per_port + (1 if i < shippers_remainder else 0)

    # Generate name if not provided
    if args.name:
        name = args.name
    else:
        name = f"custom_{args.consignees}-{args.shippers}_{args.periods}"

    config = InstanceConfig(
        name=name,
        consignees_per_dryport=consignees_per_dryport,
        shippers_per_dryport=shippers_per_dryport,
        periods=args.periods,
        seed=args.seed,
    )

    generator = ScenarioGenerator(base_dir)
    output_path = generator.generate_and_save(config, output_dir)

    print(f"Generated instance: {output_path}")
    return output_path


def main():
    """Main entry point."""
    args = parse_args()

    if args.command is None:
        print("Error: No command specified. Use -h for help.")
        sys.exit(1)

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "from-config":
        cmd_from_config(args)
    else:
        print(f"Error: Unknown command '{args.command}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
