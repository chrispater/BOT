import React, { useEffect, useState } from 'react'
import { Card, EmptyState } from '../components/Primitives'

function UserItem({ user, onApprove, onToggleAdmin }) {
  const isApproved = user.account_status === 'approved'
  const statusLabel = user.account_status === 'approved' ? 'Approved' : user.account_status === 'rejected' ? 'Rejected' : 'Pending'

  return (
    <div className="user-item">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 5 }}>{user.username}</div>
          <div>
            <span className={`user-tag ${isApproved ? 'approved' : 'pending'}`}>{statusLabel}</span>
            {user.is_admin && <span className="user-tag is-admin">Admin</span>}
          </div>
        </div>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>#{user.id}</span>
      </div>
      <div className="user-actions">
        <button className={isApproved ? 'revoke' : 'approve'} onClick={() => onApprove(user.id, !isApproved)}>
          {isApproved ? 'Revoke Access' : 'Approve'}
        </button>
        <button className={user.is_admin ? 'revoke' : 'admin-on'} onClick={() => onToggleAdmin(user.id, !user.is_admin)}>
          {user.is_admin ? 'Remove Admin' : 'Make Admin'}
        </button>
      </div>
    </div>
  )
}

export default function AdminPage({ api, setError, setSuccess }) {
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)

  const loadUsers = async () => {
    setLoading(true)
    try {
      const res = await api.get('/admin/users')
      setUsers(res.data.users || [])
    } catch (e) {
      setError('Failed to load users')
    }
    setLoading(false)
  }

  useEffect(() => { loadUsers() }, [])

  const approveUser = async (userId, makeApproved) => {
    try {
      await api.post(`/admin/users/${userId}/approve`, { action: makeApproved ? 'approve' : 'reject' })
      setSuccess(makeApproved ? 'User approved' : 'User access revoked')
      loadUsers()
    } catch (e) {
      setError('Failed to update user')
    }
  }

  const toggleAdmin = async (userId, makeAdmin) => {
    try {
      await api.post(`/admin/users/${userId}/permissions`, { is_admin: makeAdmin })
      setSuccess(makeAdmin ? 'Admin privileges granted' : 'Admin privileges removed')
      loadUsers()
    } catch (e) {
      setError('Failed to update permissions')
    }
  }

  if (loading) return <div className="loading"><div className="spinner"></div></div>

  const pending = users.filter(u => u.account_status !== 'approved')
  const approved = users.filter(u => u.account_status === 'approved')

  return (
    <>
      <Card title="User Management" right={
        <button onClick={loadUsers} style={{ padding: '6px 12px', background: 'var(--accent-dim)', border: '1px solid rgba(233,69,96,0.3)', borderRadius: 8, color: 'var(--accent)', cursor: 'pointer', fontSize: 12, fontWeight: 600 }}>
          Refresh
        </button>
      }>
        {users.length === 0 && <EmptyState>No users found</EmptyState>}

        {pending.length > 0 && (
          <>
            <div style={{ fontSize: 12, color: 'var(--accent-gold)', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Pending Approval ({pending.length})
            </div>
            {pending.map(user => <UserItem key={user.id} user={user} onApprove={approveUser} onToggleAdmin={toggleAdmin} />)}
            <div style={{ height: 12 }} />
          </>
        )}

        {approved.length > 0 && (
          <>
            <div style={{ fontSize: 12, color: 'var(--accent-green)', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Approved ({approved.length})
            </div>
            {approved.map(user => <UserItem key={user.id} user={user} onApprove={approveUser} onToggleAdmin={toggleAdmin} />)}
          </>
        )}
      </Card>

      <Card title="Admin Guide">
        <ul className="strategy-list">
          <li>New registrations require approval before accessing the bot</li>
          <li>Approved users can start the bot and manage settings</li>
          <li>Admin users can approve/revoke other users and grant admin rights</li>
          <li>Revoking access immediately prevents the user from logging in</li>
        </ul>
      </Card>
    </>
  )
}
