resource "aws_efs_file_system" "this" {
  creation_token = "${var.name_prefix}-qdrant"
  encrypted      = true

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-qdrant"
  })
}

resource "aws_security_group" "efs" {
  name        = "${var.name_prefix}-efs-sg"
  description = "Allow NFS from ECS tasks to EFS"
  vpc_id      = var.vpc_id
  tags        = var.tags
}

resource "aws_security_group_rule" "efs_ingress_nfs" {
  type              = "ingress"
  description       = "NFS from ECS tasks"
  from_port         = 2049
  to_port           = 2049
  protocol          = "tcp"
  security_group_id = aws_security_group.efs.id
  source_security_group_id = var.security_group_ids[0]
}

resource "aws_efs_mount_target" "this" {
  count           = length(var.subnet_ids)
  file_system_id  = aws_efs_file_system.this.id
  subnet_id       = var.subnet_ids[count.index]
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_access_point" "qdrant" {
  file_system_id = aws_efs_file_system.this.id

  posix_user {
    uid = 1000
    gid = 1000
  }

  root_directory {
    path = "/qdrant-storage"
    creation_info {
      owner_uid   = 1000
      owner_gid   = 1000
      permissions = "0755"
    }
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-qdrant-ap"
  })
}
